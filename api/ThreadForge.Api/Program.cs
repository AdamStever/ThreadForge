using ThreadForge.Api;

var builder = WebApplication.CreateBuilder(args);

// Path to the SQLite feature store the Python pipeline writes. Override via
// appsettings ("FeatureStore:DbPath") or the env var FeatureStore__DbPath.
var dbPath = builder.Configuration["FeatureStore:DbPath"] ?? "threadforge.db";
builder.Services.AddSingleton(new FeatureStoreReader(dbPath));

var app = builder.Build();

// Require a valid X-API-Key on every request except the open liveness paths.
// The expected key comes from configuration (Auth:ApiKey), never from code.
app.UseMiddleware<ApiKeyMiddleware>();

// --- read-only endpoints over the feature store ---

app.MapGet("/", () => Results.Ok(new { service = "ThreadForge.Api", status = "ok" }));

app.MapGet("/health", () => Results.Ok(new { status = "ok" }));

// all recorded runs
app.MapGet("/runs", (FeatureStoreReader store) => Results.Ok(store.ListRuns()));

// summary of one run
app.MapGet("/runs/{id:long}", (long id, FeatureStoreReader store) =>
{
    var summary = store.Summarize(id);
    return summary is null
        ? Results.NotFound(new { error = $"no run with id {id}" })
        : Results.Ok(summary);
});

// the signal names recorded for a run/channel
app.MapGet("/runs/{id:long}/signals",
    (long id, FeatureStoreReader store, string channel = FeatureStoreReader.DefaultChannel) =>
        Results.Ok(store.SignalNames(id, channel)));

// one signal's time series for a run/channel
app.MapGet("/runs/{id:long}/signals/{name}",
    (long id, string name, FeatureStoreReader store, string channel = FeatureStoreReader.DefaultChannel) =>
        Results.Ok(store.ReadSignal(id, name, channel)));

// the raw input stream for a run/channel
app.MapGet("/runs/{id:long}/stream",
    (long id, FeatureStoreReader store, string channel = FeatureStoreReader.DefaultChannel) =>
        Results.Ok(store.ReadStream(id, channel)));

app.Run();
