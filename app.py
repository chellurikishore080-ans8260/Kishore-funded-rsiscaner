"""Kishore Funded Pairs - RSI Long + RSI Short scanner & backtest (Binance Spot, Streamlit).

Only the funded pairs you list are scanned/backtested. Same logic as the main RSI scanner:
  RSI Long : RSI <= 31.5 buy at candle close | exit RSI >= 57 | stop RSI <= 16
  RSI Short: RSI >= 69 sell at candle close | exit RSI <= 37 | stop RSI >= 89
A trade is a WIN only if the price PnL is > 0 (one single win definition).
All times are IST and are the candle CLOSE time (entry/exit happen at candle close).
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------- EDIT HERE ----------------
DEFAULT_PAIRS = "BTC, ETH, LTC, BCH, XRP, ADA, DOGE, SOL"   # funded pairs (POL, DOT, LINK removed)
ALIASES = {"MATIC": "POL"}               # Binance renamed MATIC to POL
LONG_IN, LONG_SL, LONG_TP = 31.5, 16, 57
SHORT_IN, SHORT_SL, SHORT_TP = 69, 89, 37
MAX_LOSS = 1300         # Rs: a trade is stopped as soon as its loss reaches this (all timeframes except 4h)
MAX_LOSS_4H = 3000      # Rs: same, for the 4h timeframe only
RSI_LEN, ST_LEN, ST_MULT = 14, 10, 3.0
# -------------------------------------------
HIST_OPTS = [1000, 2000, 3000, 5000, 8000, 10000]   # candles per timeframe for the backtest
TFS = ["5m", "15m", "30m", "1h", "4h"]
TF_MS = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000}
HOSTS = ["https://data-api.binance.vision", "https://api.binance.com",
         "https://api1.binance.com", "https://api2.binance.com", "https://api3.binance.com"]
TZ = "Asia/Kolkata"
FMT = "%d-%b-%Y %H:%M"


# ---------------- DATA ----------------
_HOST = {"good": None}      # host that worked last (tried first next time)
ERRORS = {}                 # (symbol, tf) -> last error text, shown in the app


def api(path, params=None):
    order = ([_HOST["good"]] if _HOST["good"] else []) + [h for h in HOSTS if h != _HOST["good"]]
    last = None
    for h in order:
        try:
            r = requests.get(h + path, params=params, timeout=8)
            if r.status_code == 200:
                _HOST["good"] = h
                return r.json()
            last = f"{h.split('//')[1]} HTTP {r.status_code}"
        except Exception as e:
            last = f"{h.split('//')[1]} {type(e).__name__}"
    raise RuntimeError(last)


def parse_pairs(text):
    out, seen = [], set()
    for tok in re.split(r"[,\s;]+", text.upper()):
        tok = re.sub(r"[/_\-]", "", tok.strip())
        if not tok:
            continue
        base = re.sub(r"USDT?$", "", tok)
        if base in ALIASES:
            tok = ALIASES[base] + tok[len(base):]
        if tok.endswith("USDT"):
            sym = tok
        elif tok.endswith("USD"):
            sym = tok + "T"
        else:
            sym = tok + "USDT"
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def rsi_calc(c, n=RSI_LEN):
    d = c.diff()
    au = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + au / ad)


def supertrend(df, n=ST_LEN, m=ST_MULT):
    h, l, c = df.high.values, df.low.values, df.close.values
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values
    hl2 = (h + l) / 2
    ub, lb = hl2 + m * atr, hl2 - m * atr
    fub, flb = ub.copy(), lb.copy()
    d, line = np.ones(len(c)), lb.copy()
    for i in range(1, len(c)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if d[i - 1] == 1:
            d[i] = -1 if c[i] < flb[i] else 1
        else:
            d[i] = 1 if c[i] > fub[i] else -1
        line[i] = flb[i] if d[i] == 1 else fub[i]
    return line, d


def fetch_history(sym, tf, total):
    """Up to `total` closed candles, paged backwards 1000 at a time."""
    rows, end = [], None
    while len(rows) < total:
        lim = min(1000, total - len(rows))
        p = {"symbol": sym, "interval": tf, "limit": lim}
        if end:
            p["endTime"] = end
        try:
            raw = api("/api/v3/klines", p)
        except Exception:
            if rows:                       # keep what we already got instead of losing everything
                break
            raise
        if not raw:
            break
        rows = raw + rows
        end = raw[0][0] - 1
        if len(raw) < lim:
            break
    return rows


@st.cache_resource
def _store():
    return {}


def load_one(sym, tf, total, ttl, store):
    key = (sym, tf, total)
    hit = store.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    try:
        raw = fetch_history(sym, tf, total)
        df = pd.DataFrame([r[:6] for r in raw],
                          columns=["ot", "open", "high", "low", "close", "vol"]).astype(float)
        df["cm"] = df.ot + TF_MS[tf]                       # candle CLOSE time (ms)
        df = df[df.cm <= time.time() * 1000].drop_duplicates("ot").reset_index(drop=True)
        if len(df) < 60:
            ERRORS[(sym, tf)] = "too little data"
            return None
        df["rsi"] = rsi_calc(df.close)
        line, d = supertrend(df)
        # bearish Supertrend rejects price: wick touches the line, candle closes below it
        df["rej"] = (d == -1) & (df.high.values >= line) & (df.close.values < line)
        df["t"] = pd.to_datetime(df.cm, unit="ms", utc=True).dt.tz_convert(TZ)
    except Exception as e:
        ERRORS[(sym, tf)] = str(e)[:120]
        return None
    store[key] = (time.time(), df)
    return df


def load_many(pairs, total, ttl, progress=None):
    store = _store()                       # get the cache in the main thread, not in workers
    now = time.time()
    for k in [k for k, v in store.items() if now - v[0] > 1800]:
        store.pop(k, None)                 # drop old data so big histories don't pile up in memory
    res = {}
    with ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(load_one, s, t, total, ttl, store): (s, t) for s, t in pairs}
        for n, f in enumerate(as_completed(futs), 1):
            res[futs[f]] = f.result()
            if progress:
                progress(n / len(pairs))   # called from the main thread only
    return {p: res[p] for p in pairs}


# ---------------- BACKTEST ----------------
def _trade(sym, tf, side, j, i, result, reason, df, ts, xp=None):
    cl, r, rej = df.close.values, df.rsi.values, df.rej.values
    e = cl[j]
    x = cl[i] if xp is None else xp                    # xp = exact stop price for a max-loss stop
    pct = (x / e - 1) * 100 if side == "Long" else (e - x) / e * 100
    entry = f"🟢 RSI ≤ {LONG_IN}" if side == "Long" else f"🔴 RSI ≥ {SHORT_IN}"
    st_at_exit = ("✅ reject" if rej[i] else "no") if side == "Long" else "–"
    return {"Coin": sym.replace("USDT", ""), "TF": tf, "Side": side,
            "Entry Time": ts[j], "Entry Price": e, "Exit Time": ts[i], "Exit Price": x,
            "Entry Logic": entry, "Exit Logic": reason, "Bearish ST": st_at_exit,
            "RSI In": r[j], "RSI Out": r[i], "Candles": i - j,
            "PnL %": pct, "_ts": df.cm.values[j], "_state": result}


def run_bt(df, sym, tf, strict=False, notional=0.0, max_loss=0.0):
    r, rej, cl = df.rsi.values, df.rej.values, df.close.values
    op, hi, lo = df["open"].values, df["high"].values, df["low"].values
    sl = max_loss / notional if (max_loss and notional) else None      # max loss as a fraction of price
    ts = df.t.dt.strftime(FMT).values                         # formatted once per series (fast for 10000 candles)
    out, pos = [], None
    for i in range(len(df)):
        if np.isnan(r[i]):
            continue
        if pos is None:                                   # entry at candle close
            if r[i] <= LONG_IN:
                pos = ("Long", i)
            elif r[i] >= SHORT_IN:
                pos = ("Short", i)
            continue
        side, j = pos
        if sl is not None:                                # max-loss stop: checked on every candle high/low
            if side == "Long":
                lvl = cl[j] * (1 - sl)
                hit = lo[i] <= lvl
                fill = op[i] if op[i] <= lvl else lvl     # a gap beyond the stop fills at the open
            else:
                lvl = cl[j] * (1 + sl)
                hit = hi[i] >= lvl
                fill = op[i] if op[i] >= lvl else lvl
            if hit:
                out.append(_trade(sym, tf, side, j, i, "CLOSED", f"🛑 Max loss stop (₹{max_loss:,.0f})",
                                  df, ts, fill))
                pos = None
                continue
        reason = None
        if side == "Long":
            if r[i] <= LONG_SL:
                reason = f"🛑 RSI stop (≤ {LONG_SL})"
            elif r[i] >= LONG_TP:
                reason = f"🎯 RSI target (≥ {LONG_TP})"
        else:
            if r[i] >= SHORT_SL:
                reason = f"🛑 RSI stop (≥ {SHORT_SL})"
            elif r[i] <= SHORT_TP:
                reason = f"🎯 RSI target (≤ {SHORT_TP})"
        if reason and strict and "stop" not in reason:
            if not (cl[i] > cl[j] if side == "Long" else cl[i] < cl[j]):
                reason = None                              # strict: price must be in profit too
        if reason:
            out.append(_trade(sym, tf, side, j, i, "CLOSED", reason, df, ts))
            pos = None
    if pos:
        out.append(_trade(sym, tf, pos[0], pos[1], len(df) - 1, "OPEN", "⏳ Still open", df, ts))
    return out


def make_trades(data, notional, strict, limits=None):
    limits = limits or {}
    rows = []
    for (s, t), df in data.items():
        if df is not None:
            rows += run_bt(df, s, t, strict, notional, limits.get(t, 0))
    if not rows:
        return pd.DataFrame()
    tr = pd.DataFrame(rows)
    tr["PnL ₹"] = notional * tr["PnL %"] / 100
    tr["Result"] = np.where(tr._state == "OPEN", "OPEN", np.where(tr["PnL ₹"] > 0, "WIN", "LOSS"))
    return tr.drop(columns="_state")


# ---------------- STATS ----------------
def sstats(g):
    n = len(g)
    w = int((g["PnL ₹"] > 0).sum())
    return {"Trades": n, "Wins": w, "Losses": n - w,
            "Win %": round(w / n * 100, 1) if n else None,
            "PnL ₹": round(g["PnL ₹"].sum()) if n else 0}


def coin_side_table(v):
    rows = []
    for coin, g in v.groupby("Coin"):
        L, S, T = sstats(g[g.Side == "Long"]), sstats(g[g.Side == "Short"]), sstats(g)
        lw, sw = L["Win %"], S["Win %"]
        if L["Trades"] and S["Trades"]:
            best = "Long" if (lw, L["PnL ₹"]) > (sw, S["PnL ₹"]) else "Short"
        else:
            best = "Long" if L["Trades"] else "Short" if S["Trades"] else "-"
        rows.append({"Coin": coin,
                     "Long Trades": L["Trades"], "Long Wins": L["Wins"], "Long Losses": L["Losses"],
                     "Long Win %": lw, "Long PnL ₹": L["PnL ₹"],
                     "Short Trades": S["Trades"], "Short Wins": S["Wins"], "Short Losses": S["Losses"],
                     "Short Win %": sw, "Short PnL ₹": S["PnL ₹"],
                     "Total": T["Trades"], "Wins": T["Wins"], "Losses": T["Losses"],
                     "Win %": T["Win %"], "PnL ₹": T["PnL ₹"], "Better side": best})
    return pd.DataFrame(rows).sort_values("Win %", ascending=False)


def coin_tf_table(v):
    rows = []
    for (coin, tf), g in v.groupby(["Coin", "TF"]):
        T, L, S = sstats(g), sstats(g[g.Side == "Long"]), sstats(g[g.Side == "Short"])
        rows.append({"Coin": coin, "TF": tf, **T,
                     "Long W/L": f"{L['Wins']}/{L['Losses']}", "Short W/L": f"{S['Wins']}/{S['Losses']}",
                     "Worst Loss %": round(g["PnL %"].min(), 2)})
    out = pd.DataFrame(rows)
    out["_o"] = out.TF.map({t: k for k, t in enumerate(TFS)})
    return out.sort_values(["Coin", "_o"]).drop(columns="_o")


def exit_logic_table(v):
    rows = [{"Side": side, "Exit Logic": reason, **sstats(g)}
            for (side, reason), g in v.groupby(["Side", "Exit Logic"])]
    return pd.DataFrame(rows)


def px(x):
    return f"{x:,.2f}" if x >= 100 else f"{x:.6g}"


def hl(row):
    if row.get("Loss #") == 1:
        s = "background-color:rgba(255,0,0,0.6);font-weight:bold"
    elif row["Result"] == "LOSS":
        s = "background-color:rgba(255,0,0,0.22)"
    elif row["Result"] == "WIN":
        s = "background-color:rgba(0,200,0,0.17)"
    else:
        s = "background-color:rgba(150,150,150,0.25)"
    return [s] * len(row)


def trade_view(g):
    g = g.sort_values("_ts").reset_index(drop=True).copy()
    g.insert(0, "#", range(1, len(g) + 1))
    g["Loss #"] = np.nan
    cl = g[(g.Result == "LOSS")]
    g.loc[cl.index, "Loss #"] = cl["PnL ₹"].rank(method="min")     # 1 = biggest loss
    # logic sits right next to its price: Entry Price -> Entry Logic, Exit Price -> Exit Logic
    g["Entry Logic"] = g["Entry Logic"] + g["RSI In"].map(lambda x: f"  (RSI {x:.1f})")
    g["Exit Logic"] = g["Exit Logic"] + g["RSI Out"].map(lambda x: f"  (RSI {x:.1f})")
    cols = ["#", "Side", "Entry Time", "Entry Price", "Entry Logic",
            "Exit Time", "Exit Price", "Exit Logic", "Bearish ST",
            "Candles", "PnL %", "PnL ₹", "Result", "Loss #"]
    return g[cols].style.apply(hl, axis=1).format(
        {"Entry Price": px, "Exit Price": px,
         "PnL %": "{:+.2f}", "PnL ₹": "{:+,.0f}", "Loss #": "{:.0f}"}, na_rep="")


# ---------------- UI ----------------
def settings():
    pairs_txt = st.text_area("Funded pairs (comma tho veru cheyyi)", st.session_state.get("pairs", DEFAULT_PAIRS),
                             help="BTC, ETHUSDT, SOL/USDT ila edi rasina chalu. Binance Spot USDT pairs.")
    st.session_state["pairs"] = pairs_txt
    return parse_pairs(pairs_txt)


def backtest_tab(pairs):
    c1, c2 = st.columns(2)
    tfs = c1.multiselect("Timeframes", TFS, default=TFS)
    hist = c2.select_slider("History (candles per TF)", HIST_OPTS, value=3000)
    if hist >= 8000:
        st.caption(f"⏱ {hist} candles ki data techukodaniki konchem time padutundi (progress bar chupistundi).")
    c3, c4 = st.columns(2)
    cap = c3.number_input("Capital ₹", 10000, 10_000_000, 100000, 10000)
    per = c4.slider("Capital per trade %", 10, 100, 50, 10)
    strict = st.checkbox("Strict: RSI target tappudu price kuda profit lo unte matrame exit")
    c5, c6 = st.columns(2)
    ml = c5.number_input("Max loss ₹ per trade (4h tappa anni TF)", 0, 1_000_000, MAX_LOSS, 100,
                         help="Trade loss ee amount ki cherina ventane trade stop. 0 = off.")
    ml4 = c6.number_input("Max loss ₹ per trade (4h)", 0, 1_000_000, MAX_LOSS_4H, 100,
                          help="4h timeframe ki matrame. 0 = off.")
    notional = cap * per / 100
    if notional:
        st.caption(f"Trade size ₹{notional:,.0f} → max loss = price {ml / notional * 100:.2f}% (4h: {ml4 / notional * 100:.2f}%) "
                   "entry price nundi viruddha disha lo velithe stop.")
    limits = {t: (ml4 if t == "4h" else ml) for t in TFS}
    if st.button("▶ Run backtest", type="primary", use_container_width=True):
        if not pairs or not tfs:
            st.warning("Pairs, timeframes select cheyyi.")
            return
        prog = st.progress(0.0, text="Binance nunchi data techukuntunnanu...")
        data = load_many([(s, t) for s in pairs for t in tfs], hist, 600,
                         progress=lambda x: prog.progress(x, text=f"Data load avutondi... {int(x * 100)}%"))
        prog.empty()
        bad = [f"{s}-{t}" for (s, t), d in data.items() if d is None]
        st.session_state["bt"] = {"tr": make_trades(data, notional, strict, limits), "bad": bad,
                                  "cap": cap, "hist": hist, "data": data}
    bt = st.session_state.get("bt")
    if not bt:
        st.info("Run backtest nokku.")
        return
    tr, cap = bt["tr"], bt["cap"]
    if bt["bad"]:
        st.warning("Data raaledu (skip): " + ", ".join(bt["bad"]))
        errs = sorted({ERRORS.get(tuple(b.split("-")), "") for b in bt["bad"]} - {""})
        if errs:
            st.caption("Reason: " + " | ".join(errs[:3]))
    if tr.empty:
        st.warning("Trades levu.")
        return
    v = tr[tr.Result != "OPEN"]
    n_open = int((tr.Result == "OPEN").sum())
    if v.empty:
        st.warning("Closed trades levu.")
        return

    t1, t2, t3 = st.tabs(["📊 Summary", "📦 Coin × TF boxes", "📅 Data range"])
    with t1:
        summary(v, cap, n_open)
    with t2:
        boxes(tr)
    with t3:
        rng = []
        for (s, t), d in bt["data"].items():
            if d is not None:
                rng.append({"Coin": s.replace("USDT", ""), "TF": t, "Candles": len(d),
                            "From": d.t.iloc[0].strftime(FMT), "To": d.t.iloc[-1].strftime(FMT)})
        with st.container(border=True):
            st.subheader("Backtest data range (IST)")
            st.dataframe(pd.DataFrame(rng), hide_index=True)


def summary(v, cap, n_open):
    T, L, S = sstats(v), sstats(v[v.Side == "Long"]), sstats(v[v.Side == "Short"])
    with st.container(border=True):
        st.subheader("Overall")
        st.markdown(
            f"**Total trades: {T['Trades']}** &nbsp;·&nbsp; ✅ Wins: **{T['Wins']}** &nbsp;·&nbsp; "
            f"❌ Losses: **{T['Losses']}** &nbsp;·&nbsp; Win %: **{T['Win %']}%**\n\n"
            f"**Net PnL: ₹{T['PnL ₹']:,}** ({T['PnL ₹'] / cap * 100:+.2f}% of capital) "
            f"&nbsp;·&nbsp; Open (not counted): {n_open}\n\n"
            f"🟢 **Long:** {L['Wins']}W / {L['Losses']}L = **{L['Win %'] if L['Trades'] else '-'}%** "
            f"&nbsp;·&nbsp; ₹{L['PnL ₹']:,}\n\n"
            f"🔴 **Short:** {S['Wins']}W / {S['Losses']}L = **{S['Win %'] if S['Trades'] else '-'}%** "
            f"&nbsp;·&nbsp; ₹{S['PnL ₹']:,}")
        st.caption("Win = price PnL > 0. Loss = PnL ≤ 0. RSI target hit ayina price loss lo unte adi LOSS ga count.")

    cs = coin_side_table(v)
    with st.container(border=True):
        st.subheader("Highlights")
        mt = st.slider("Min trades (side ki)", 1, 10, 3)
        for side in ("Long", "Short"):
            q = cs[cs[f"{side} Trades"] >= mt]
            if q.empty:
                st.info(f"{side}: {mt}+ trades unna coin ledu.")
                continue
            b = q.sort_values([f"{side} Win %", f"{side} Trades"], ascending=False).iloc[0]
            w = q.sort_values([f"{side} Win %", f"{side} Trades"]).iloc[0]
            st.success(f"{'🟢' if side == 'Long' else '🔴'} {side} best: **{b.Coin}** — {b[f'{side} Win %']}% "
                       f"({b[f'{side} Wins']}W/{b[f'{side} Losses']}L)")
            st.error(f"{side} worst: **{w.Coin}** — {w[f'{side} Win %']}% "
                     f"({w[f'{side} Wins']}W/{w[f'{side} Losses']}L)")
        wt = v.loc[v["PnL ₹"].idxmin()]
        if wt["PnL ₹"] < 0:
            st.error(f"🔻 Biggest loss: {wt.Coin} {wt.TF} {wt.Side} — {wt['PnL %']:+.2f}% (₹{wt['PnL ₹']:,.0f}) • "
                     f"{wt['Entry Time']} @ {px(wt['Entry Price'])} → {wt['Exit Time']} @ {px(wt['Exit Price'])}")

    with st.container(border=True):
        st.subheader("Coin-wise: Long vs Short")
        st.dataframe(cs, hide_index=True)
    with st.container(border=True):
        st.subheader("Exit logic wise")
        st.dataframe(exit_logic_table(v), hide_index=True)
    ct = coin_tf_table(v)
    with st.container(border=True):
        st.subheader("Coin × Timeframe (wins / losses)")
        st.dataframe(ct, hide_index=True)
    with st.container(border=True):
        st.subheader("Win % — Coin × Timeframe")
        pv = ct.pivot(index="Coin", columns="TF", values="Win %")
        st.dataframe(pv[[t for t in TFS if t in pv.columns]])
        pv = ct.pivot(index="Coin", columns="TF", values="PnL ₹")
        st.caption("PnL ₹")
        st.dataframe(pv[[t for t in TFS if t in pv.columns]])


def boxes(tr):
    coins = sorted(tr.Coin.unique())
    sel = st.multiselect("Coins", coins, default=coins)
    if not sel:
        st.info("Kanisam oka coin select cheyyi.")
        return
    st.caption(f"Entry logic: 🟢 Long RSI ≤ {LONG_IN} · 🔴 Short RSI ≥ {SHORT_IN}  |  Exit logic: "
               f"🎯 Long RSI ≥ {LONG_TP} · 🛑 Long RSI ≤ {LONG_SL} · "
               f"🎯 Short RSI ≤ {SHORT_TP} · 🛑 Short RSI ≥ {SHORT_SL}")
    for coin in sel:                                           # coin-wise -> timeframe-wise
        c = tr[tr.Coin == coin]
        cc = c[c.Result != "OPEN"]
        if cc.empty:
            head = f"🪙 {coin} — closed trades levu"
        else:
            T = sstats(cc)
            head = f"🪙 {coin} — {T['Trades']} trades · {T['Wins']}W/{T['Losses']}L · {T['Win %']}% · ₹{T['PnL ₹']:,}"
        with st.expander(head, expanded=len(sel) == 1):
            for tf in TFS:
                g = c[c.TF == tf]
                if g.empty:
                    continue
                closed = g[g.Result != "OPEN"]
                with st.container(border=True):
                    st.subheader(f"{coin} • {tf}")
                    if closed.empty:
                        st.caption("Closed trades levu.")
                    else:
                        T, L, S = (sstats(closed), sstats(closed[closed.Side == "Long"]),
                                   sstats(closed[closed.Side == "Short"]))
                        st.markdown(
                            f"**Trades {T['Trades']}** · ✅ Wins **{T['Wins']}** · ❌ Losses **{T['Losses']}** · "
                            f"Win **{T['Win %']}%** · PnL **₹{T['PnL ₹']:,}**\n\n"
                            f"🟢 Long {L['Wins']}W/{L['Losses']}L ({L['Win %'] if L['Trades'] else '-'}%) · "
                            f"🔴 Short {S['Wins']}W/{S['Losses']}L ({S['Win %'] if S['Trades'] else '-'}%)")
                    st.dataframe(trade_view(g), hide_index=True, height=min(420, 60 + 36 * len(g)))
    st.caption("🛑 Max loss stop: candle high/low tho check, Exit Price = stop price; Exit Time = aa candle close time. "
               "Dark red = ee box lo biggest loss (Loss #1). Grey = inka open (stats lo lekkaledu). "
               "Times = candle close, IST.")


def live_tab(pairs):
    tfs = st.multiselect("Timeframes", TFS, default=["5m", "15m", "1h", "4h"], key="ltf")
    fresh = st.slider("Signal age (candles)", 1, 5, 2, key="lf")

    @st.fragment(run_every=60)
    def body():
        if not pairs or not tfs:
            return
        data = load_many([(s, t) for s in pairs for t in tfs], 400, 50)
        sig, snap = [], []
        for (s, t), df in data.items():
            if df is None:
                continue
            r = df.rsi.values
            snap.append({"Coin": s.replace("USDT", ""), "TF": t, "Price": px(df.close.values[-1]),
                         "RSI now": round(r[-1], 1)})
            for age in range(fresh):
                i = len(df) - 1 - age
                side = "🟢 Long" if r[i] <= LONG_IN else "🔴 Short" if r[i] >= SHORT_IN else None
                if side:
                    sig.append({"Signal Time (IST)": df.t.dt.strftime(FMT).values[i], "Coin": s.replace("USDT", ""),
                                "TF": t, "Signal": side, "Price": px(df.close.values[i]), "RSI": round(r[i], 1),
                                "Status": "FRESH" if age == 0 else f"{age} candle ago"})
                    break
        st.caption(f"Auto-refresh 60s • {pd.Timestamp.now(tz=TZ):%H:%M:%S} IST")
        with st.container(border=True):
            st.subheader("Signals")
            if sig:
                st.dataframe(pd.DataFrame(sig).sort_values("Signal Time (IST)", ascending=False), hide_index=True)
            else:
                st.info(f"RSI ≤ {LONG_IN} / ≥ {SHORT_IN} signal ippudu ledu.")
        with st.container(border=True):
            st.subheader("RSI now")
            if snap:
                st.dataframe(pd.DataFrame(snap), hide_index=True)

    body()


def main():
    st.set_page_config(page_title="Funded Pairs RSI", page_icon="🏦", layout="wide")
    st.title("🏦 Funded Pairs — RSI Scanner")
    with st.expander("📖 Logic"):
        st.markdown(
            f"- **RSI Long** — RSI ≤ {LONG_IN} → buy at candle close. Stop: RSI ≤ {LONG_SL}. "
            f"Exit: RSI ≥ {LONG_TP}.\n"
            f"- **RSI Short** — RSI ≥ {SHORT_IN} → sell. Stop: RSI ≥ {SHORT_SL}. Exit: RSI ≤ {SHORT_TP}.\n"
            "- Oka coin + timeframe ki okkate position (exit ayyaka matrame next entry).\n"
            f"- **Max loss stop** — trade loss ₹{MAX_LOSS:,} ki (4h lo ₹{MAX_LOSS_4H:,}) cherite trade stop. "
            "Candle high/low tho check chestam, exit price = stop price (gap ayite open price).\n"
            "- Backtest history 10000 candles varaku (TF ki). Prathi trade lo entry/exit time, price, "
            "entry logic, exit logic (RSI target / stop) chupistundi.\n"
            "- **Win** = price PnL > 0 matrame. RSI target hit ayyi price loss lo unte adi LOSS.\n"
            "- RSI trades ki price stop ledu, RSI level ye stop, kabatti loss pedda undochu.\n"
            "- Times IST, candle close time.")
    pairs = settings()
    st.caption("Pairs: " + ", ".join(p.replace("USDT", "") for p in pairs))
    view = st.radio("View", ["📊 Backtest", "🔴 Live"], horizontal=True, label_visibility="collapsed")
    if view.endswith("Backtest"):
        backtest_tab(pairs)
    else:
        live_tab(pairs)               # only runs (and fetches data) when selected


if __name__ == "__main__":
    main()
