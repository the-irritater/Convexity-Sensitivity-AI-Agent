"""Dashboard Page 5: Bond Risk Lab — Gamified Platform"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import load_bond_portfolio
from src.part6_bond_risk_lab import (ScenarioChallenge, QUIZ_QUESTIONS, ACHIEVEMENTS, RISK_PROFILES)

st.set_page_config(page_title="Bond Risk Lab", page_icon="🎮", layout="wide")
st.title("🎮 Bond Risk Lab — Gamified Training Platform")

df = load_bond_portfolio()
challenge = ScenarioChallenge(df)

# Session state for game
if 'game_score' not in st.session_state:
    st.session_state.game_score = 0
    st.session_state.scenarios_played = 0
    st.session_state.correct_answers = 0
    st.session_state.achievements = []

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Scenario Challenge", "📝 Quiz Mode", "🏅 Achievements", "📊 Risk Profile"])

with tab1:
    st.markdown("### Predict the Portfolio Impact!")
    
    difficulty = st.select_slider("Difficulty", ["Easy", "Medium", "Hard"])
    
    if st.button("🎲 Generate New Scenario", use_container_width=True):
        scenario = challenge.generate_scenario(difficulty)
        st.session_state.current_scenario = scenario
    
    if 'current_scenario' in st.session_state:
        s = st.session_state.current_scenario
        st.markdown(f"""
        ### Scenario: Yield curve shifts by **{s['shock_bps']:+d} bps**
        
        > **Question**: Will the portfolio **gain** or **lose** value? By approximately how much?
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            user_dir = st.radio("Direction", ["Gain 📈", "Loss 📉"])
        with col2:
            user_pct = st.number_input("Estimated P&L (%)", 0.0, 30.0, 5.0, 0.5)
        
        if st.button("Submit Answer", use_container_width=True):
            direction = 'gain' if 'Gain' in user_dir else 'loss'
            result = challenge.evaluate_prediction(s, direction, user_pct)
            
            st.session_state.game_score += result['score']
            st.session_state.scenarios_played += 1
            
            if result['direction_correct']:
                st.success(f"✅ Correct! The portfolio would {'gain' if s['actual_pnl'] > 0 else 'lose'} value.")
                st.session_state.correct_answers += 1
            else:
                st.error(f"❌ Incorrect. The portfolio would {'gain' if s['actual_pnl'] > 0 else 'lose'} value.")
            
            st.info(f"""
            **Actual Results:**
            - P&L: ₹{s['actual_pnl']/1e5:+.2f} Lakhs ({s['actual_pct']:+.4f}%)
            - Duration Effect: ₹{s['duration_effect']/1e5:+.2f} L
            - Convexity Effect: ₹{s['convexity_effect']/1e5:+.2f} L
            - Points earned: **+{result['score']}**
            """)
    
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 Score", st.session_state.game_score)
    c2.metric("🎯 Scenarios Played", st.session_state.scenarios_played)
    c3.metric("✅ Correct", st.session_state.correct_answers)

with tab2:
    st.markdown("### Fixed Income Knowledge Quiz")
    
    for i, q in enumerate(QUIZ_QUESTIONS):
        with st.expander(f"Q{q['id']}. [{q['difficulty']}] {q['question']}"):
            answer = st.radio(
                "Select your answer:",
                q['options'],
                key=f"quiz_{q['id']}"
            )
            
            if st.button("Check Answer", key=f"check_{q['id']}"):
                if answer == q['options'][q['correct']]:
                    st.success("✅ Correct!")
                    st.session_state.game_score += 10
                else:
                    st.error(f"❌ Incorrect. The correct answer is: **{q['options'][q['correct']]}**")
                st.info(f"💡 {q['explanation']}")

with tab3:
    st.markdown("### Achievement Badges")
    
    cols = st.columns(4)
    for i, (aid, ach) in enumerate(ACHIEVEMENTS.items()):
        with cols[i % 4]:
            unlocked = aid in st.session_state.achievements
            status = "🔓" if unlocked else "🔒"
            st.markdown(f"""
            <div style="background: {'linear-gradient(135deg, #f7971e, #ffd200)' if unlocked else '#333'};
                        padding: 1rem; border-radius: 12px; margin: 0.5rem 0; text-align: center;
                        color: {'#1a1a2e' if unlocked else '#888'};">
                <div style="font-size: 2rem;">{ach['icon']}</div>
                <div style="font-weight: 700; font-size: 0.9rem;">{ach['name']}</div>
                <div style="font-size: 0.75rem; margin-top: 0.3rem;">{ach['description'][:50]}...</div>
                <div style="font-size: 0.8rem; margin-top: 0.3rem;">{status} +{ach['points']} pts</div>
            </div>
            """, unsafe_allow_html=True)

with tab4:
    st.markdown("### Risk Tolerance Profile")
    
    for name, profile in RISK_PROFILES.items():
        with st.expander(f"{profile['icon']} {name} — Target Duration: {profile['target_duration']} years"):
            st.markdown(f"""
            - **Description**: {profile['description']}
            - **Target Duration**: {profile['target_duration']} years
            - **Max Convexity**: {profile['max_convexity']}
            - **Recommended Sectors**: {', '.join(profile['sectors'])}
            """)
    
    # Interactive duration sensitivity
    st.markdown("### 📐 Interactive Sensitivity Calculator")
    shock_bps = st.slider("Yield Shock (bps)", -300, 300, 100, 25)
    dy = shock_bps / 10000.0
    
    total_mv = df['MarketValue_INR'].sum()
    weights = df['MarketValue_INR'] / total_mv
    port_dur = (weights * df['ModifiedDuration']).sum()
    port_conv = (weights * df['Convexity']).sum()
    
    dur_effect = -port_dur * dy * total_mv
    conv_effect = 0.5 * port_conv * dy**2 * total_mv
    total_pnl = dur_effect + conv_effect
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Duration Effect", f"₹{dur_effect/1e5:+,.0f} L")
    c2.metric("Convexity Effect", f"₹{conv_effect/1e5:+,.0f} L")
    c3.metric("Total P&L", f"₹{total_pnl/1e5:+,.0f} L",
              delta=f"{total_pnl/total_mv*100:+.2f}%")
