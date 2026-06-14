# Linux I/O for Robotics — Course Build Spec (READ FIRST)

You are authoring ONE self-contained chapter HTML file for a polished, offline, dark-themed
learning site (read on desktop + iPad). This file is the contract. The **gold-standard example**
is `../pid/01-plant.html` (a flat course, same depth as this one) and the **style guide** is
`../embedded-mastery/_AUTHORING_GUIDE.md` — open and mirror them. Where this spec and the embedded
guide differ on *paths/classes*, THIS spec wins (this course lives at `io/`, depth 1, like `pid/`).

Course identity: **Linux I/O for Robotics** — how I/O actually works in Linux, taught
intuition → math → code → real case → practice, with the **Jetson AGX Orin USV** as the running
real-world example. `data-course="io"`, `data-track="main"`. Accent: copper-orange. Icon 🔀.

---

## Non-negotiable rules (the validator enforces these — `node deploy/validate.mjs --course=io --lang=en`)
1. COMPLETE valid HTML: starts `<!DOCTYPE html>`, ends `</html>`. No `TODO`, no `...` inside tags, no `lorem`.
2. **Vendored assets only — never a CDN or `http(s)://` URL.** Paths are relative from `io/` → `../assets/...`; brand/home → `../index.html`.
3. `<body class="course-io" data-course="io" data-track="main" data-chapter="<ID>">` with the exact ID from the registry below.
4. Leave the nav empty: `<nav class="sidebar" id="sidebar"></nav>` and `<nav class="chap-nav" id="chap-nav"></nav>`. `app.js` fills them.
5. Every `<h2>` has a **unique kebab-case `id`** and a `<span class="sec-no">N</span>`. Non-ref pages: **≥6 `<h2>`** (aim 6–8). The glossary (ref page): ≥3.
6. **Exactly the 3 exercise types** on every non-ref page: one `data-ex="quiz"`, one `data-ex="numeric"` (with `data-answer`, optional `data-tol`), one `data-ex="reveal"`. Each `.exercise` has a **unique `data-id`** (prefix `ioNN-`). Quiz must include a `data-correct="true"` option. Glossary: ≥1 exercise.
7. Inside `<code>`/`<pre>`: escape `<` as `&lt;` and `&` as `&amp;`.
8. Scripts at the very end, **`app.js` last among libraries**; any sim init script goes AFTER `app.js`.

## Exact page skeleton (copy; change only title, ids, content)
```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>CHAPTER TITLE — Linux I/O for Robotics</title>
<link rel="stylesheet" href="../assets/katex/katex.min.css">
<link rel="stylesheet" href="../assets/vendor/atom-one-dark.min.css">
<link rel="stylesheet" href="../assets/styles.css">
</head>
<body class="course-io" data-course="io" data-track="main" data-chapter="01-io-model">
<div id="read-progress"></div>
<header class="topbar">
  <button class="menu-btn" id="menu-btn" aria-label="Open chapter menu">☰</button>
  <a class="brand" href="../index.html"><span class="dot"></span> Linux&nbsp;I/O</a>
  <span class="spacer"></span>
  <span class="track-pill">I/O · Ch 01</span>
  <span class="done-chip" id="done-chip">In progress</span>
</header>
<div class="layout">
  <nav class="sidebar" id="sidebar"></nav>
  <div class="sb-backdrop" id="sb-backdrop"></div>
  <main class="main">
    <article class="content">
      <section class="chapter-hero">
        <div class="eyebrow">Linux I/O · Chapter 01 · Foundations</div>
        <h1>Chapter Title</h1>
        <p>One-paragraph hook: why this matters for a robot/USV.</p>
        <div class="meta"><span>⏱ ~35 min</span><span>🧠 concept</span><span>⚡ hands-on</span></div>
      </section>

      <!-- sections: each <h2 id> + .sec-no, with .layer badges, callouts, math-box, codeblock, mermaid -->

      <nav class="chap-nav" id="chap-nav"></nav>
    </article>
  </main>
</div>
<button id="to-top" aria-label="Back to top">↑</button>
<script src="../assets/vendor/highlight.min.js"></script>
<script src="../assets/katex/katex.min.js"></script>
<script src="../assets/katex/contrib/auto-render.min.js"></script>
<script src="../assets/vendor/mermaid.min.js"></script>
<script src="../assets/app.js"></script>
<!-- if this chapter has a sim, add AFTER app.js:
<script src="../assets/io-sim.js"></script>
<script>EM.ioDemo.uartFrame("sim-uart");</script> -->
</body>
</html>
```

## Components (already styled — use these classes)
- **Layers:** `<div class="layer"><span class="step">①</span> Intuition <small>build the picture</small></div>` (steps ①②③④⑤ = intuition/math/code/real/practice).
- **Callouts:** `<div class="callout intuition|key|math|real|warn|tip"><div class="c-title"><span class="ico">💡</span> Title</div><p>…</p></div>` (icons: 💡 intuition, 🔑 key, ∑ math, 🏭 real, ⚠️ warn, 🛠️ tip).
- **Math (KaTeX):** inline `$…$`, display `$$…$$`; headline equations in `<div class="math-box">$$ … $$</div>`. Include real formulas where they genuinely help (throughput, latency, bit-time, sampling, bus-load, BDP). A few solid formulas per chapter; never force math where it doesn't belong.
- **Code:** `<div class="codeblock"><div class="code-head">file.c <span class="lang">C</span></div><pre><code class="language-c">…</code></pre></div>`. Langs: `language-c`, `language-bash`, `language-python`. ESCAPE `<` and `&`.
- **Tables:** `<div class="tbl-wrap"><table>…</table></div>`.
- **Diagrams:** `<div class="mermaid">flowchart LR …</div>` — ≥1 per chapter. Short labels, `<br/>` for breaks. Do NOT call mermaid.initialize.
- **Exercises:** quiz/numeric/reveal exactly as in `../pid/01-plant.html` §7. Unique `data-id` per exercise.
- **Recap:** end each chapter with `<div class="callout key"><div class="c-title"><span class="ico">🔑</span> Chapter recap</div><p>…</p></div>` right before `<nav class="chap-nav" id="chap-nav">`.

## Tone & depth
Sharp senior engineer teaching a capable peer who knows the shell but is fuzzy on kernel/driver/networking
internals. Direct, concrete, technically RIGOROUS (real syscalls, real `/sys` paths, real commands, real
numbers). Build intuition before formalism. ~1500–2400 words prose + code/diagrams. Every command must be
real and correct. Use the Jetson AGX Orin USV for "real case" callouts. Later chapters may assume earlier ones.

## Interactive engine — `assets/io-sim.js` (call AFTER app.js)
Each demo: a `<div class="sim"><div class="sim-title"><span>EMOJI</span> Title</div><canvas id="ID"></canvas></div>`
then `<script>EM.ioDemo.NAME("ID");</script>`. Available demos and where they belong:

| Chapter | `<canvas id>` | Call | Shows |
|---|---|---|---|
| 05-buses-enumeration | `sim-bus` | `EM.ioDemo.busBandwidth("sim-bus")` | log-scale bus bandwidths vs a draggable "required throughput" line |
| 06-serial-tty | `sim-uart` | `EM.ioDemo.uartFrame("sim-uart")` | one UART frame waveform; baud/byte/parity/stop sliders; bit-time + byte-rate |
| 07-interrupts-dma-mmio | `sim-move` | `EM.ioDemo.dataMovement("sim-move")` | CPU ops for PIO vs IRQ(+FIFO coalescing) vs DMA over a transfer |
| 08-multiplexing-epoll | `sim-mux` | `EM.ioDemo.ioMultiplex("sim-mux")` | thread-per-fd vs epoll cost as #fds scales |
| 09-networking-io | `sim-net` | `EM.ioDemo.netLink("sim-net")` | latency-vs-load for Ethernet (full-duplex) vs WiFi (shared, jitter) |
| 10-can-socketcan | `sim-can` | `EM.ioDemo.canLoad("sim-can")` | CAN bus-load gauge vs bitrate / frame-rate / payload |

Only these 6 chapters embed a sim; the others rely on Mermaid + tables. The sim is ONE concept per chapter — wrap it with a short "what to notice" `callout key` after it.

---

## The 13 pages (filename · id · objective · sections · sim)

**01 · `01-io-model.html` · `01-io-model` — The Linux I/O Model.**
The syscall boundary, file descriptors, `open/read/write/close/lseek/ioctl`, user vs kernel space, the VFS,
"everything is a file," partial reads, buffered (stdio) vs raw, `errno`. Sections: why I/O is the hard part
of robotics; user vs kernel space + the syscall (mermaid app→libc→syscall→driver→device); fds & the fd table
(0/1/2); the core syscalls + partial reads (C + Python); everything-is-a-file & the VFS; buffered vs
unbuffered; throughput vs syscall overhead (math: effective rate, amortization). Real case: IMU at 200 Hz over
serial. No sim. Exercises: quiz(read() partial-return), numeric(effective throughput), reveal(robust read loop w/ EINTR+partial).

**02 · `02-devices-sysfs.html` · `02-devices-sysfs` — Devices as Files: /dev & sysfs.**
char vs block, major/minor (`dev_t`), devtmpfs, `/sys` (class/bus/devices), reading/writing sysfs attrs, the
unified device model (bus↔driver↔device), ioctl. Sections: device files & types; major/minor encoding (math:
`dev_t` bitfields); how /dev is populated (devtmpfs); a tour of `/sys`; reading & writing attributes (GPIO,
link speed, latency_timer); the device model & binding; ioctl for char devices. Real case: find your sensor
under `/sys`, read link speed, toggle a GPIO. No sim. Exercises: quiz(char vs block), numeric(major/minor or device-number math), reveal(sysfs read/write commands).

**03 · `03-udev-naming.html` · `03-udev-naming` — udev & Persistent Naming.**
uevents, udevd, rule syntax (match vs assign), matching by `idVendor/idProduct/serial`, `SYMLINK+=`,
`/dev/serial/by-id`, by-path, predictable network interface names + `.link`, `RUN/MODE/GROUP`. Sections: the
renumbering problem (math: n! orderings); kernel uevent → udev flow; rule anatomy; finding match keys
(`udevadm info`); writing a stable-symlink rule; testing (`udevadm test/trigger`); by-id/by-path & net naming.
Real case: `/dev/gps`, `/dev/imu`, pinned `eth` name on the USV. No sim. Exercises: quiz(match vs assign key),
numeric(n! device orderings), reveal(write a udev rule).

**04 · `04-drivers-modules-firmware.html` · `04-drivers-modules-firmware` — Drivers, Modules & Firmware.**
kernel vs userspace drivers, module lifecycle, `insmod/modprobe/modinfo/depmod/lsmod`, params + `/etc/modprobe.d`,
the firmware loader (`request_firmware`, `/lib/firmware`, error -2), in-tree vs out-of-tree (DKMS, kABI/kernel
coupling), userspace drivers (libusb/spidev/i2c-dev/UIO/VFIO), reading `dmesg`. Real case: NXP `mlan`/`moal`
out-of-tree driver + firmware blob (taught generally). No sim. Exercises: quiz(why out-of-tree breaks on kernel
upgrade), numeric(e.g. version/firmware sizing), reveal(modprobe with params + dmesg check).

**05 · `05-buses-enumeration.html` · `05-buses-enumeration` — Buses & Enumeration (+ Device Tree).**
Enumerable (PCIe/USB) vs declared (I²C/SPI/SDIO/CAN → device tree) buses; PCIe config space/BAR/`lspci`; USB
descriptors/`lsusb -t`; I²C/SPI addressing; SDIO; device tree (nodes, `compatible`, `reg`, `status`, pinmux),
overlays, `jetson-io`. **Sim: `EM.ioDemo.busBandwidth("sim-bus")`.** Real case: AGX Orin — PCIe auto-enumerates,
SDIO (the MAYA WiFi) must be declared in the DT. Math: I²C 7-bit = 128 addresses, bandwidth, BDF. Exercises:
quiz(which bus needs DT), numeric(I²C address space or bus throughput), reveal(read a DT node / lspci -t).

**06 · `06-serial-tty.html` · `06-serial-tty` — Serial & the TTY Layer.**
UART frame anatomy, the TTY stack (core/line-discipline/driver), canonical vs raw, termios (`cfmakeraw`,
`cfsetspeed`, `VMIN/VTIME`), baud math, flow control (RTS/CTS, XON/XOFF), latency (FTDI `latency_timer`,
VMIN/VTIME), `ttyUSB` vs `ttyTHS`. **Sim: `EM.ioDemo.uartFrame("sim-uart")`.** Real case: GPS/IMU at 200 Hz
without lag. Math: bit time $=1/\text{baud}$, 8N1 efficiency $=8/10$, byte rate, frame time. Exercises:
quiz(8N1 efficiency / framing), numeric(bytes/s or frame time at a baud), reveal(termios raw-mode setup in C).

**07 · `07-interrupts-dma-mmio.html` · `07-interrupts-dma-mmio` — How Data Moves: Interrupts, DMA & MMIO.**
PIO vs IRQ vs DMA; MMIO & memory-mapped registers (`mmap`, `volatile`); interrupts (IRQ line, ISR, top/bottom
half, threaded IRQ); `/proc/interrupts` + IRQ affinity; DMA (descriptors, coherent vs streaming, cache);
coalescing & throughput-vs-latency; NAPI polling under IRQ storms. **Sim: `EM.ioDemo.dataMovement("sim-move")`.**
Real case: high-rate LiDAR over Ethernet — DMA + coalescing keep CPU free for the 20 Hz control loop. Math:
interrupt rate, CPU cost per IRQ, DMA efficiency. Exercises: quiz(why DMA), numeric(interrupts/s & CPU%),
reveal(`/proc/interrupts` + set IRQ affinity).

**08 · `08-multiplexing-epoll.html` · `08-multiplexing-epoll` — Multiplexing I/O: poll, epoll & io_uring.**
blocking vs non-blocking (`O_NONBLOCK`), thread-per-fd and why it doesn't scale, `select`/`poll` (O(n)),
`epoll` (O(ready), edge vs level), `io_uring` (SQ/CQ rings, true async), the C10k problem, the event-loop
pattern. **Sim: `EM.ioDemo.ioMultiplex("sim-mux")`.** Real case: one thread reading GPS+IMU+LiDAR+CAN
concurrently. Math: O(n) vs O(ready), context-switch cost, scaling. Exercises: quiz(edge vs level trigger),
numeric(scan cost / scaling), reveal(epoll event-loop skeleton).

**09 · `09-networking-io.html` · `09-networking-io` — Networking I/O: NIC to Socket.**
packet path NIC→driver→stack→socket (mermaid); the socket API as fds; interfaces, addresses, routing table,
metrics, **default-route theft**; two-NIC coexistence (Ethernet sensor + WiFi control); WiFi as a shared
half-duplex medium (latency/jitter); buffers/backpressure (`SO_RCVBUF`, bufferbloat); UDP vs TCP for sensor
data. **Sim: `EM.ioDemo.netLink("sim-net")`.** Real case: AGX Orin — Ethernet LiDAR + WiFi SSH; keep heavy
data off WiFi; the default-route gotcha. Math: bandwidth-delay product, throughput vs latency, Little's law.
Exercises: quiz(default-route theft), numeric(BDP or throughput), reveal(read `ip route` / non-blocking UDP socket).

**10 · `10-can-socketcan.html` · `10-can-socketcan` — CAN & SocketCAN.**
CAN physical (differential, arbitration by ID priority); frame anatomy + bit-stuffing; **SocketCAN** (CAN as a
network socket — `AF_CAN`); bring-up (`ip link set can0 type can bitrate … restart-ms …`); bus load & sizing;
termination (the 60 Ω test) & sample point; error counters, bus-off, `restart-ms`; CAN-FD; NMEA 2000 for marine.
**Sim: `EM.ioDemo.canLoad("sim-can")`.** Real case: USV actuators + marine instruments (NMEA 2000 @ 250 k).
Math: bits/frame, bus load %, sample point. Exercises: quiz(arbitration winner: lowest ID wins), numeric(bus
load %), reveal(SocketCAN bring-up + `candump`).

**11 · `11-realtime-robust-io.html` · `11-realtime-robust-io` — Real-Time & Robust I/O.**
what "real-time" means (deadlines, worst-case ≠ average, jitter); latency sources (scheduling, IRQ, page
faults, power states); PREEMPT_RT, `SCHED_FIFO`/rtprio, `mlockall`; CPU/IRQ isolation (`isolcpus`, affinity);
robust I/O loop (`EINTR`, partial read/write, `EAGAIN`, timeouts, reconnection); durable logging (page cache,
`fsync`, `O_DIRECT`, ring buffers, SD-card wear); watchdogs & failsafe. Real case: the 20 Hz PID loop must
never miss; log mission data without blocking the loop; WiFi-drop failsafe. No sim. Math: jitter/deadline,
worst-case latency, fsync cost. Exercises: quiz(avg vs worst-case), numeric(deadline/jitter budget), reveal(robust read loop with EINTR + timeout).

**12 · `12-capstone-robot-io.html` · `12-capstone-robot-io` — Capstone: Architecting a Robot's I/O.**
Integrate the course into the USV's I/O architecture: map every sensor → bus → driver → fd (mermaid of all I/O
on the AGX Orin); interface-role assignment (WiFi=control, Ethernet=heavy sensors, USB/serial=low-rate,
CAN=actuators); the bring-up chain as persistent config (DT + firmware + udev + netplan + systemd ordering);
a unified epoll sensor-manager design; health monitoring & a pre-flight check; the failure-handling matrix
(WiFi drop / USB disconnect / CAN bus-off); a bandwidth budget across buses; whole-course recap. No new sim.
Exercises: quiz(interface role assignment), numeric(bus bandwidth budget), reveal(systemd unit + pre-flight script, or epoll sensor-manager skeleton).

**G · `glossary.html` · `glossary` — Glossary (EN/中).** REF page: ≥3 `<h2>`, ≥1 exercise. Grouped term tables
(EN term · 中文 · one-line definition): Core I/O · Devices & drivers · Buses · Networking · CAN · Real-time.
One quiz exercise at the end. Keep `data-track="main"`. Still include the standard skeleton, scripts, empty navs.

---

## Final self-check before returning a chapter
- [ ] `<!DOCTYPE html>` … `</html>`, no TODO/placeholder/external URL.
- [ ] `class="course-io"` + correct `data-course/track/chapter`.
- [ ] All assets `../assets/...`; brand `../index.html`; `app.js` last; sim script (if any) after app.js.
- [ ] Empty `#sidebar` and `#chap-nav`.
- [ ] ≥6 `<h2>` (glossary ≥3), every one a unique kebab-case id with `.sec-no`.
- [ ] 3 exercises (quiz+numeric+reveal; glossary ≥1), unique `data-id` (`ioNN-…`), quiz has `data-correct="true"`, numeric has `data-answer`.
- [ ] All `<`/`&` escaped inside `<code>`; ≥1 mermaid diagram; recap `callout key` before chap-nav.
- [ ] If a sim chapter: correct `<canvas id>` + `io-sim.js` + exact `EM.ioDemo.NAME("id")` call.
