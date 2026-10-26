import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# পেজ সেটআপ - প্রফেশনাল ডার্ক ও চওড়া লেআউট
st.set_page_config(page_title="Pro AI Market Terminal", page_icon="⚡", layout="wide")

# কাস্টম সিএসএস স্টাইল (Professional Dashboard Look)
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; background-color: #4CAF50; color: white; border-radius: 8px; font-weight: bold; }
    .metric-box { padding: 15px; background-color: #1f293d; border-radius: 10px; border: 1px solid #3b4252; }
    </style>
    """, unsafe_allowed_html=True)

st.title("⚡ Pro AI Market Terminal & Live Alert System")
st.write("অ্যাডভান্সড মেশিন লার্নিং ও ইন্ডিকেটর ভিত্তিক রিয়েল-টাইম মার্কেট অ্যানালাইসিস টার্মিনাল।")

# সাইডবারে ইনপুট অপশন
st.sidebar.header("🔧 কন্ট্রোল প্যানেল")
ticker = st.sidebar.text_input("স্টক সিম্বল (যেমন: RELIANCE.NS, TATAMOTORS.NS):", "RELIANCE.NS")
timeframe = st.sidebar.selectbox("টাইমফ্রেম (Timeframe):", ["5m", "15m", "1h", "1d"], index=3)

# অ্যানালাইসিস শুরু
try:
    # টাইমফ্রেম অনুযায়ী ডাটা ডাউনলোড করা
    if timeframe in ["5m", "15m", "1h"]:
        df = yf.download(ticker, period="1mo", interval=timeframe)
    else:
        df = yf.download(ticker, period="3y", interval="1d")
        
    if df.empty:
        st.error("❌ সঠিক স্টক কোড বা সিম্বল লিখুন।")
    else:
        df = df.copy()
        
        # ১. প্রফেশনাল ইন্ডিকেটর ১: RSI
        df['Price_Diff'] = df['Close'].diff()
        gain = df['Price_Diff'].clip(lower=0)
        loss = -df['Price_Diff'].clip(upper=0)
        ema_gain = gain.ewm(com=13, adjust=False).mean()
        ema_loss = loss.ewm(com=13, adjust=False).mean()
        rs = ema_gain / ema_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ২. প্রফেশনাল ইন্ডিকেটর ২: EMA 9 & EMA 21 (Crossover Strategy)
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        
        # ৩. প্রفেশনাল ইন্ডিকেটর ৩: Volatility (দামের তীব্রতা)
        df['Return'] = df['Close'].pct_change()
        df['Volatility'] = df['High'] - df['Low']
        
        # ৪. AI মডেলের জন্য টার্গেট (দাম কি পরবর্তী ক্যান্ডেলে বাড়বে?)
        df['Price_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)

        # AI মডেল ট্রেনিং
        X = df[['Return', 'Volatility', 'RSI', 'EMA_9', 'EMA_21']]
        y = df['Price_Up']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # রিয়েল-টাইম ডাটা দিয়ে প্রেডিকশন
        latest_data = np.array([[
            df['Return'].iloc[-1].item(), 
            df['Volatility'].iloc[-1].item(), 
            df['RSI'].iloc[-1].item(),
            df['EMA_9'].iloc[-1].item(),
            df['EMA_21'].iloc[-1].item()
        ]])
        prediction = model.predict(latest_data)
        prob = model.predict_proba(latest_data)[0]

        # শেষ ভ্যালুগুলো বের করা
        last_close = df['Close'].iloc[-1].item()
        rsi_val = df['RSI'].iloc[-1].item()
        ema9_val = df['EMA_9'].iloc[-1].item()
        ema21_val = df['EMA_21'].iloc[-1].item()

        # ড্যাশবোর্ড লেআউট তৈরি (৩টি কলাম)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='metric-box'>", unsafe_allowed_html=True)
            st.metric(label="💰 সর্বশেষ লাইভ প্রাইস", value=f"₹{last_close:.2f}")
            st.markdown("</div>", unsafe_allowed_html=True)
            
        with col2:
            st.markdown("<div class='metric-box'>", unsafe_allowed_html=True)
            st.metric(label="📊 RSI (14) ভ্যালু", value=f"{rsi_val:.2f}", 
                      delta="অতিরিক্ত কেনা (Overbought)" if rsi_val > 70 else "অতিরিক্ত বিক্রি (Oversold)" if rsi_val < 30 else "স্বাভাবিক জোন")
            st.markdown("</div>", unsafe_allowed_html=True)
            
        with col3:
            st.markdown("<div class='metric-box'>", unsafe_allowed_html=True)
            trend_signal = "⚡ BULLISH Crossover" if ema9_val > ema21_val else "⚠️ BEARISH Crossover"
            st.metric(label="📈 Trend (EMA 9/21)", value=trend_signal)
            st.markdown("</div>", unsafe_allowed_html=True)

        # এআই প্রেডিকশন ও লাইভ অ্যালার্ট সেকশন
        st.subheader("🎯 এআই লাইভ অ্যালার্ট ও সিগন্যাল")
        
        confidence = prob[1] if prediction == 1 else prob[0]
        
        if prediction == 1:
            st.success(f"📈 [🚨 LIVE ALERT] MARKET TREND: UP (দাম বাড়ার সম্ভাবনা রয়েছে) | এআই কনফিডেন্স: {confidence*100:.1f}%")
            st.info("💡 সাজেস্টেড অ্যাকশন: বাই (Buy) পজিশন খোঁজা যেতে পারে। অবশ্যই স্টপ-লস বজায় রাখুন।")
        else:
            st.error(f"📉 [🚨 LIVE ALERT] MARKET TREND: DOWN (দাম কমার সম্ভাবনা রয়েছে) | এআই কনফিডেন্স: {confidence*100:.1f}%")
            st.info("💡 সাজেস্টেড অ্যাকশন: সেল (Sell) পজিশন বা সতর্কতা অবলম্বন করুন।")

        # প্রফেশনাল ক্যান্ডেলস্টিক চার্ট (Candlestick Chart)
        st.subheader("📊 অ্যাডভান্সড টেকনিক্যাল চার্ট (Interactive Chart)")
        fig = go.Figure(data=[go.Candlestick(x=df.index[-50:],
                        open=df['Open'].tail(50), high=df['High'].tail(50),
                        low=df['Low'].tail(50), close=df['Close'].tail(50),
                        name="Market Price")])
        fig.add_trace(go.Scatter(x=df.index[-50:], y=df['EMA_9'].tail(50), name='EMA 9', line=dict(color='orange', width=1.5)))
        fig.add_trace(go.Scatter(x=df.index[-50:], y=df['EMA_21'].tail(50), name='EMA 21', line=dict(color='blue', width=1.5)))
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"একটি যান্ত্রিক ত্রুটি ঘটেছে: {e}")
