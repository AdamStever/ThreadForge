using System.Security.Cryptography;
using System.Text;

namespace ThreadForge.Api;

/// <summary>
/// Minimal API-key authentication. Every request must carry a matching
/// <c>X-API-Key</c> header except the open liveness paths (<c>/</c>, <c>/health</c>).
///
/// The expected key is read from configuration (<c>Auth:ApiKey</c>) — never
/// hardcoded. Supply it out-of-band:
///   - dev:  dotnet user-secrets set "Auth:ApiKey" "&lt;key&gt;"
///   - prod: environment variable  Auth__ApiKey=&lt;key&gt;
///
/// Fails closed: if no key is configured, protected endpoints are rejected
/// rather than left open.
/// </summary>
public sealed class ApiKeyMiddleware
{
    public const string HeaderName = "X-API-Key";

    private static readonly HashSet<string> OpenPaths =
        new(StringComparer.OrdinalIgnoreCase) { "/", "/health" };

    private readonly RequestDelegate _next;
    private readonly ILogger<ApiKeyMiddleware> _logger;

    public ApiKeyMiddleware(RequestDelegate next, ILogger<ApiKeyMiddleware> logger)
    {
        _next = next;
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context, IConfiguration config)
    {
        var path = context.Request.Path.Value ?? "/";
        if (OpenPaths.Contains(path))
        {
            await _next(context);
            return;
        }

        var configuredKey = config["Auth:ApiKey"];
        if (string.IsNullOrEmpty(configuredKey))
        {
            _logger.LogWarning(
                "Auth:ApiKey is not configured — rejecting request to {Path}. "
                + "Set it via user-secrets (dev) or the Auth__ApiKey env var (prod).",
                path);
            await Deny(context);
            return;
        }

        var provided = context.Request.Headers[HeaderName].ToString();
        if (string.IsNullOrEmpty(provided) || !FixedTimeEquals(provided, configuredKey))
        {
            await Deny(context);
            return;
        }

        await _next(context);
    }

    /// <summary>Constant-time string comparison to avoid leaking the key via timing.</summary>
    private static bool FixedTimeEquals(string a, string b)
    {
        var ba = Encoding.UTF8.GetBytes(a);
        var bb = Encoding.UTF8.GetBytes(b);
        return ba.Length == bb.Length && CryptographicOperations.FixedTimeEquals(ba, bb);
    }

    private static Task Deny(HttpContext context)
    {
        context.Response.StatusCode = StatusCodes.Status401Unauthorized;
        return context.Response.WriteAsJsonAsync(new { error = "missing or invalid API key" });
    }
}
