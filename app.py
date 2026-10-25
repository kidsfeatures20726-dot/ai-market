import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Market Predictor", page_icon="📈", layout="centered")
st.title("🤖 AI শেয়ার মার্কেট প্রেডিকশন সাইট")
st.write("মোবাইলে এআই দিয়ে জানুন আগামীকালের মার্কেট ট্রেন্ড!")

ticker = st.text_input("শেয়ারের নাম লিখুন (যেমন: RELIANCE.NS, TCS.NS):", "RELIANCE.NS")

if st.button("প্রেডিক্ট করুন"):
    with st.spinner('এআই ডাটা অ্যানালাইসিস করছে...'):
        try:
            df = yf.download(ticker, period="3y")
            if df.empty:
                st.error("ভুল নাম দিয়েছেন! সঠিক কোড লিখুন।")
            else:
                # ডেটা ফরম্যাটের নতুন সমস্যা দূর করার জন্য
                df = df.copy()
                
                df['Return'] = df['Close'].pct_change()
                df['Volatility'] = df['High'] - df['Low']
                df['Price_Diff'] = df['Close'].diff()
                
                gain = df['Price_Diff'].clip(lower=0)
                loss = -df['Price_Diff'].clip(upper=0)
                ema_gain = gain.ewm(com=13, adjust=False).mean()
                ema_loss = loss.ewm(com=13, adjust=False).mean()
                rs = ema_gain / ema_loss
                df['RSI'] = 100 - (100 / (1 + rs))
                df['Price_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
                df.dropna(inplace=True)

                X = df[['Return', 'Volatility', 'RSI']]
                y = df['Price_Up']
                model = RandomForestClassifier(n_estimators=100, random_state=1)
                model.fit(X, y)

                latest_data = np.array([[df['Return'].iloc[-1].item(), df['Volatility'].iloc[-1].item(), df['RSI'].iloc[-1].item()]])
                prediction = model.predict(latest_data)

                # সিঙ্গেল ভ্যালু বের করার জন্য .item() ব্যবহার করা হয়েছে যাতে format ভুল না দেখায়
                last_close_val = df['Close'].iloc[-1].item()
                rsi_val = df['RSI'].iloc[-1].item()

                st.subheader(f"📊 {ticker} এর বর্তমান অবস্থা:")
                st.metric(label="সর্বশেষ দাম (Last Close)", value=f"₹{last_close_val:.2f}")
                st.metric(label="RSI ইন্ডিকেটর", value=f"{rsi_val:.2f}")

                st.subheader("🎯 এআই ফাইনাল প্রেডিকশন:")
                if prediction == 1:
                    st.success("📈 MARKET TREND: UP (আগামীকাল দাম বাড়তে পারে)")
                else:
                    st.error("📉 MARKET TREND: DOWN (আগামীকাল দাম কমতে পারে)")
                
                st.subheader("📈 বিগত ১ মাসের ক্লোজিং প্রাইস চার্ট:")
                st.line_chart(df['Close'].tail(30))
        except Exception as e:
            st.error(f"একটি সমস্যা হয়েছে: {e}")
