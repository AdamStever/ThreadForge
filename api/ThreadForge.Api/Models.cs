namespace ThreadForge.Api;

/// <summary>One recorded pipeline run.</summary>
public record RunInfo(long RunId, string Source, string StartedAt);

/// <summary>A single (timestamp, value) point of a stream or signal series.</summary>
public record SeriesPoint(string Timestamp, double? Value);

/// <summary>A compact summary of one run.</summary>
public record RunSummary(
    long RunId,
    string Source,
    string StartedAt,
    IReadOnlyList<string> Channels,
    IReadOnlyList<string> Signals,
    long StreamPoints,
    long SignalRows,
    string? TimeStart,
    string? TimeEnd);
