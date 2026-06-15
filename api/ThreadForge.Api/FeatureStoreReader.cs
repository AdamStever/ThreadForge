using Microsoft.Data.Sqlite;

namespace ThreadForge.Api;

/// <summary>
/// Read-only access to a ThreadForge SQLite feature store — the same database the
/// Python pipeline writes with <c>run_detection.py --store</c>. The .NET service
/// is a thin serving layer over that shared store: Python detects and writes,
/// this reads and exposes the results over HTTP. It never mutates the database
/// (opened read-only).
///
/// Schema (see src/threadforge/data/store.py):
///   runs(id, source, started_at)
///   stream_values(id, run_id, timestamp, channel, value)
///   signal_scores(id, run_id, timestamp, channel, signal_name, value)
/// </summary>
public sealed class FeatureStoreReader
{
    public const string DefaultChannel = "value";

    private readonly string _dbPath;

    public FeatureStoreReader(string dbPath) => _dbPath = dbPath;

    /// <summary>Open the store read-only, or null if the database file is absent.</summary>
    private SqliteConnection? Open()
    {
        if (!File.Exists(_dbPath))
            return null;

        var connectionString = new SqliteConnectionStringBuilder
        {
            DataSource = _dbPath,
            Mode = SqliteOpenMode.ReadOnly,
        }.ToString();

        var conn = new SqliteConnection(connectionString);
        conn.Open();
        return conn;
    }

    public IReadOnlyList<RunInfo> ListRuns()
    {
        var runs = new List<RunInfo>();
        using var conn = Open();
        if (conn is null)
            return runs;

        using var cmd = conn.CreateCommand();
        cmd.CommandText = "SELECT id, source, started_at FROM runs ORDER BY id";
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
            runs.Add(new RunInfo(reader.GetInt64(0), reader.GetString(1), reader.GetString(2)));
        return runs;
    }

    public RunSummary? Summarize(long runId)
    {
        using var conn = Open();
        if (conn is null)
            return null;

        string source, startedAt;
        using (var cmd = conn.CreateCommand())
        {
            cmd.CommandText = "SELECT source, started_at FROM runs WHERE id = $id";
            cmd.Parameters.AddWithValue("$id", runId);
            using var reader = cmd.ExecuteReader();
            if (!reader.Read())
                return null; // unknown run
            source = reader.GetString(0);
            startedAt = reader.GetString(1);
        }

        var channels = DistinctStrings(conn,
            "SELECT DISTINCT channel FROM stream_values WHERE run_id = $id ORDER BY channel", runId);
        var signals = DistinctStrings(conn,
            "SELECT DISTINCT signal_name FROM signal_scores WHERE run_id = $id ORDER BY signal_name", runId);
        var streamPoints = ScalarLong(conn,
            "SELECT COUNT(*) FROM stream_values WHERE run_id = $id", runId);
        var signalRows = ScalarLong(conn,
            "SELECT COUNT(*) FROM signal_scores WHERE run_id = $id", runId);

        string? timeStart = null, timeEnd = null;
        using (var cmd = conn.CreateCommand())
        {
            cmd.CommandText = "SELECT MIN(timestamp), MAX(timestamp) FROM stream_values WHERE run_id = $id";
            cmd.Parameters.AddWithValue("$id", runId);
            using var reader = cmd.ExecuteReader();
            if (reader.Read())
            {
                timeStart = reader.IsDBNull(0) ? null : reader.GetString(0);
                timeEnd = reader.IsDBNull(1) ? null : reader.GetString(1);
            }
        }

        return new RunSummary(
            runId, source, startedAt, channels, signals, streamPoints, signalRows, timeStart, timeEnd);
    }

    public IReadOnlyList<string> SignalNames(long runId, string channel = DefaultChannel)
    {
        using var conn = Open();
        if (conn is null)
            return Array.Empty<string>();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            "SELECT DISTINCT signal_name FROM signal_scores " +
            "WHERE run_id = $id AND channel = $ch ORDER BY signal_name";
        cmd.Parameters.AddWithValue("$id", runId);
        cmd.Parameters.AddWithValue("$ch", channel);
        return ReadStringColumn(cmd);
    }

    public IReadOnlyList<SeriesPoint> ReadSignal(long runId, string signalName, string channel = DefaultChannel)
    {
        using var conn = Open();
        if (conn is null)
            return Array.Empty<SeriesPoint>();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            "SELECT timestamp, value FROM signal_scores " +
            "WHERE run_id = $id AND channel = $ch AND signal_name = $name ORDER BY id";
        cmd.Parameters.AddWithValue("$id", runId);
        cmd.Parameters.AddWithValue("$ch", channel);
        cmd.Parameters.AddWithValue("$name", signalName);
        return ReadSeries(cmd);
    }

    public IReadOnlyList<SeriesPoint> ReadStream(long runId, string channel = DefaultChannel)
    {
        using var conn = Open();
        if (conn is null)
            return Array.Empty<SeriesPoint>();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            "SELECT timestamp, value FROM stream_values " +
            "WHERE run_id = $id AND channel = $ch ORDER BY id";
        cmd.Parameters.AddWithValue("$id", runId);
        cmd.Parameters.AddWithValue("$ch", channel);
        return ReadSeries(cmd);
    }

    // --- small helpers ---

    private static List<SeriesPoint> ReadSeries(SqliteCommand cmd)
    {
        var points = new List<SeriesPoint>();
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            var ts = reader.GetString(0);
            double? value = reader.IsDBNull(1) ? null : reader.GetDouble(1);
            points.Add(new SeriesPoint(ts, value));
        }
        return points;
    }

    private static List<string> ReadStringColumn(SqliteCommand cmd)
    {
        var values = new List<string>();
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
            values.Add(reader.GetString(0));
        return values;
    }

    private static List<string> DistinctStrings(SqliteConnection conn, string sql, long runId)
    {
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        cmd.Parameters.AddWithValue("$id", runId);
        return ReadStringColumn(cmd);
    }

    private static long ScalarLong(SqliteConnection conn, string sql, long runId)
    {
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        cmd.Parameters.AddWithValue("$id", runId);
        return Convert.ToInt64(cmd.ExecuteScalar());
    }
}
