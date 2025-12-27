import asyncio
import math
from collections import deque
from aiohttp import web

TCP_HOST = "0.0.0.0"
TCP_PORT = 9000

HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8000

# ====== ПАРАМЕТРЫ ФИЛЬТРА ======
FS_HZ = 250.0        # частота дискретизации (должна совпадать с ESP32)
HP_FC = 0.5          # high-pass, Гц (убрать дрейф базовой линии)
LP_FC = 25.0         # low-pass, Гц (чем меньше, тем ровнее, но тем сильнее "смягчит" QRS)
NOTCH_F0 = 50.0      # 50.0 или 60.0
NOTCH_Q = 30.0       # добротность (выше = уже вырез)

# ====== WS клиенты ======
WS_CLIENTS = set()

# небольшой хвост последних значений (для вновь подключившегося браузера)
RING_MAX = 5000
ring = []


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class ECGFilter:
    """
    Реалтайм фильтр:
      - median(3)
      - 1st order high-pass
      - biquad notch (50/60 Hz)
      - 1st order low-pass
    """
    def __init__(self, fs: float):
        self.fs = fs
        self.dt = 1.0 / fs

        # median window
        self.med = deque(maxlen=3)

        # high-pass state
        self.hp_prev_x = 0.0
        self.hp_prev_y = 0.0
        rc_hp = 1.0 / (2.0 * math.pi * HP_FC)
        self.hp_alpha = rc_hp / (rc_hp + self.dt)

        # low-pass state
        self.lp_y = 0.0
        rc_lp = 1.0 / (2.0 * math.pi * LP_FC)
        self.lp_alpha = self.dt / (rc_lp + self.dt)

        # notch biquad coefficients + state
        self._setup_notch()

        self._inited = False

    def _setup_notch(self):
        w0 = 2.0 * math.pi * (NOTCH_F0 / self.fs)
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * NOTCH_Q)

        # RBJ notch
        b0 = 1.0
        b1 = -2.0 * cos_w0
        b2 = 1.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        # normalize
        self.nb0 = b0 / a0
        self.nb1 = b1 / a0
        self.nb2 = b2 / a0
        self.na1 = a1 / a0
        self.na2 = a2 / a0

        # state (Direct Form I/II — тут DF1)
        self.nx1 = 0.0
        self.nx2 = 0.0
        self.ny1 = 0.0
        self.ny2 = 0.0

    def _median3(self, x: float) -> float:
        self.med.append(x)
        if len(self.med) < 3:
            return x
        a, b, c = self.med[0], self.med[1], self.med[2]
        return a + b + c - min(a, b, c) - max(a, b, c)

    def _highpass(self, x: float) -> float:
        # y[n] = a * (y[n-1] + x[n] - x[n-1])
        y = self.hp_alpha * (self.hp_prev_y + x - self.hp_prev_x)
        self.hp_prev_x = x
        self.hp_prev_y = y
        return y

    def _notch(self, x: float) -> float:
        # y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2
        y = (self.nb0 * x +
             self.nb1 * self.nx1 +
             self.nb2 * self.nx2 -
             self.na1 * self.ny1 -
             self.na2 * self.ny2)

        self.nx2 = self.nx1
        self.nx1 = x
        self.ny2 = self.ny1
        self.ny1 = y
        return y

    def _lowpass(self, x: float) -> float:
        # EMA low-pass: y += alpha*(x - y)
        self.lp_y = self.lp_y + self.lp_alpha * (x - self.lp_y)
        return self.lp_y

    def process(self, x: float) -> float:
        # init baseline to avoid long settling
        if not self._inited:
            self.hp_prev_x = x
            self.hp_prev_y = 0.0
            self.lp_y = x
            self._inited = True

        y = self._median3(x)
        y = self._highpass(y)
        y = self._notch(y)
        y = self._lowpass(y)
        return y


INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>ECG Live</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root { color-scheme: light dark; }
    body {
      font-family: system-ui, sans-serif;
      margin: 0;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header { padding: 12px 16px 0; }
    #status { margin-bottom: 8px; opacity: 0.8; }
    #plot {
      flex: 1;
      min-height: 200px;
      width: 100%;
    }
    .row {
      display: flex;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
    }
    input { width: 110px; }
    #pulse { font-weight: 600; }
  </style>

  <link rel="stylesheet" href="https://unpkg.com/uplot@1.6.32/dist/uPlot.min.css">
</head>
<body>
  <header>
    <h2>ECG Live (filtered on server)</h2>
    <div id="status">Connecting…</div>

    <div class="row">
      <label>Window (sec):
        <input id="winSec" type="number" min="1" max="60" step="1" value="10">
      </label>
      <label>FS (Hz):
        <input id="fsHz" type="number" min="10" max="2000" step="10" value="250">
      </label>
      <label>Pulse windows (sec):
        <input id="pulseWin" type="text" value="5,10,20">
      </label>
      <button id="clearBtn">Clear</button>
      <span id="pulse">Pulse: —</span>
    </div>
  </header>

  <div id="plot"></div>

  <script src="https://unpkg.com/uplot@1.6.32/dist/uPlot.iife.min.js"></script>
  <script>
    const statusEl = document.getElementById("status");
    const winSecEl = document.getElementById("winSec");
    const fsHzEl = document.getElementById("fsHz");
    const clearBtn = document.getElementById("clearBtn");
    const pulseWinEl = document.getElementById("pulseWin");
    const pulseEl = document.getElementById("pulse");
    const plotEl = document.getElementById("plot");

    let y = [];
    let x = [];

    function getWinSamples() {
      const fs = Number(fsHzEl.value || 250);
      const winSec = Number(winSecEl.value || 10);
      return Math.max(10, Math.floor(fs * winSec));
    }

    function rebuildX() {
      const fs = Number(fsHzEl.value || 250);
      const n = y.length;
      x = new Array(n);
      for (let i = 0; i < n; i++) {
        x[i] = (i - (n - 1)) / fs;
      }
    }

    function getPlotSize() {
      const headerHeight = document.querySelector("header").offsetHeight;
      return {
        width: window.innerWidth,
        height: Math.max(200, window.innerHeight - headerHeight)
      };
    }

    const opts = {
      ...getPlotSize(),
      scales: { x: { time: false } },
      series: [
        { label: "t (s)" },
        { label: "ecg", stroke: "#d00", width: 2 }
      ],
      axes: [
        { label: "time (s)" },
        { label: "ADC" }
      ],
    };

    let u = new uPlot(opts, [[0],[0]], plotEl);

    function updatePlot() {
      rebuildX();
      // resetScales=true и новые ссылки на массивы (slice) — чтобы точно рисовалось
      u.setData([x.slice(), y.slice()], true);
      updatePulse();
    }

    function pushValue(v) {
      const nMax = getWinSamples();
      y.push(v);
      if (y.length > nMax) y.splice(0, y.length - nMax);
    }

    clearBtn.onclick = () => {
      y = [];
      x = [];
      u.setData([[0],[0]], true);
      updatePulse();
    };

    window.addEventListener("resize", () => {
      u.setSize(getPlotSize());
      updatePlot();
    });

    function parsePulseWindows() {
      const raw = (pulseWinEl.value || "").split(",");
      const windows = raw
        .map((v) => Number(v.trim()))
        .filter((v) => Number.isFinite(v) && v > 0 && v <= 60);
      return windows.length ? windows : [5, 10, 20];
    }

    function computeBpm(samples, fs) {
      if (samples.length < fs * 2) return null;

      let sum = 0;
      for (const v of samples) sum += v;
      const mean = sum / samples.length;

      let variance = 0;
      for (const v of samples) {
        const d = v - mean;
        variance += d * d;
      }
      const std = Math.sqrt(variance / samples.length);
      const threshold = mean + std * 0.6;
      const minSamples = Math.max(1, Math.floor(fs * 0.25));

      let peaks = 0;
      let lastPeak = -minSamples;

      for (let i = 1; i < samples.length - 1; i++) {
        const v = samples[i];
        if (v > threshold && v > samples[i - 1] && v >= samples[i + 1]) {
          if (i - lastPeak >= minSamples) {
            peaks += 1;
            lastPeak = i;
          }
        }
      }

      if (peaks < 1) return null;
      const durationSec = samples.length / fs;
      return Math.round((peaks / durationSec) * 60);
    }

    function updatePulse() {
      const fs = Number(fsHzEl.value || 250);
      const windows = parsePulseWindows();
      const parts = windows.map((sec) => {
        const samples = Math.floor(sec * fs);
        if (y.length < samples) {
          return `${sec}s: —`;
        }
        const segment = y.slice(-samples);
        const bpm = computeBpm(segment, fs);
        return `${sec}s: ${bpm ?? "—"} bpm`;
      });
      pulseEl.textContent = `Pulse: ${parts.join(" | ")}`;
    }

    const wsProto = (location.protocol === "https:") ? "wss" : "ws";
    const ws = new WebSocket(`${wsProto}://${location.host}/ws`);

    let leftover = "";

    ws.onopen = () => statusEl.textContent = "WebSocket connected. Waiting for data…";
    ws.onclose = () => statusEl.textContent = "WebSocket closed.";
    ws.onerror = () => statusEl.textContent = "WebSocket error.";

    ws.onmessage = (ev) => {
      const text = leftover + ev.data;
      const lines = text.split("\n");
      leftover = lines.pop();

      for (const line of lines) {
        const s = line.trim();
        if (!s) continue;
        const v = Number(s);
        if (!Number.isFinite(v)) continue;
        pushValue(v);
      }

      updatePlot();
      statusEl.textContent = `Receiving… buffer=${y.length} samples`;
    };
  </script>
</body>
</html>
"""


async def index(request: web.Request):
    return web.Response(text=INDEX_HTML, content_type="text/html")


async def ws_handler(request: web.Request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    WS_CLIENTS.add(ws)

    # хвост буфера новому клиенту
    if ring:
        tail = "\n".join(f"{v:.2f}" for v in ring[-1000:]) + "\n"
        await ws.send_str(tail)

    try:
        async for _ in ws:
            pass
    finally:
        WS_CLIENTS.discard(ws)

    return ws


async def broadcast(chunk: str):
    if not WS_CLIENTS:
        return
    dead = []
    for ws in WS_CLIENTS:
        try:
            await ws.send_str(chunk)
        except Exception:
            dead.append(ws)
    for ws in dead:
        WS_CLIENTS.discard(ws)


async def tcp_client_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    flt = ECGFilter(fs=FS_HZ)

    buf = ""
    try:
        while True:
            data = await reader.read(4096)
            if not data:
                break

            buf += data.decode("utf-8", errors="ignore")

            if "\n" not in buf:
                # нет целой строки
                if len(buf) > 65536:
                    buf = ""
                continue

            lines = buf.split("\n")
            buf = lines.pop()  # хвост без \n

            out_lines = []
            for line in lines:
                s = line.strip()
                if not s:
                    continue
                try:
                    v = float(int(s))   # вход сырой int
                except ValueError:
                    continue

                y = flt.process(v)

                # сохраняем и рассылаем уже фильтрованный
                ring.append(y)
                if len(ring) > RING_MAX:
                    del ring[:len(ring) - RING_MAX]

                out_lines.append(f"{y:.2f}")

            if out_lines:
                await broadcast("\n".join(out_lines) + "\n")

    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def start_tcp_server(app: web.Application):
    server = await asyncio.start_server(tcp_client_handler, TCP_HOST, TCP_PORT)
    app["tcp_server"] = server
    print(f"[TCP] listening on {TCP_HOST}:{TCP_PORT}")


async def stop_tcp_server(app: web.Application):
    server = app.get("tcp_server")
    if server:
        server.close()
        await server.wait_closed()


def main():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)

    app.on_startup.append(start_tcp_server)
    app.on_cleanup.append(stop_tcp_server)

    print(f"[HTTP] serving on http://{HTTP_HOST}:{HTTP_PORT}")
    web.run_app(app, host=HTTP_HOST, port=HTTP_PORT)


if __name__ == "__main__":
    main()
