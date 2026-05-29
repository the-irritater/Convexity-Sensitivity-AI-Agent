"""
Part 6: Bond Risk Lab — Gamified Simulation Platform
=====================================================
Interactive training platform for fixed income analysts.
Features: scenario challenges, trading simulation, quiz mode,
achievements, and risk profiling.
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils import (
    load_bond_portfolio, load_monte_carlo,
    print_section_header, print_subsection,
    REPORTS_DIR
)


# ─────────────────────────────────────────────────────────────
# Achievement System
# ─────────────────────────────────────────────────────────────

ACHIEVEMENTS = {
    "duration_master": {
        "name": "Duration Master",
        "icon": "🏆",
        "description": "Correctly predict the direction of portfolio P&L for 5 consecutive scenarios",
        "points": 100
    },
    "convexity_ninja": {
        "name": "Convexity Ninja",
        "icon": "🥷",
        "description": "Identify the convexity advantage in 3 asymmetric shock scenarios",
        "points": 150
    },
    "yield_curve_wizard": {
        "name": "Yield Curve Wizard",
        "icon": "🧙",
        "description": "Correctly classify 5 yield curve movements (parallel/steepening/flattening)",
        "points": 120
    },
    "risk_manager": {
        "name": "Risk Manager",
        "icon": "🛡️",
        "description": "Successfully hedge portfolio duration within ±0.5 years of target",
        "points": 200
    },
    "perfect_score": {
        "name": "Perfect Score",
        "icon": "⭐",
        "description": "Score 100% on the fixed income quiz",
        "points": 250
    },
    "speed_trader": {
        "name": "Speed Trader",
        "icon": "⚡",
        "description": "Complete a trading simulation in under 60 seconds",
        "points": 100
    },
    "survivor": {
        "name": "Crisis Survivor",
        "icon": "🔥",
        "description": "Navigate a portfolio through a 300bps shock with less than 15% drawdown",
        "points": 180
    },
    "diversifier": {
        "name": "The Diversifier",
        "icon": "🌐",
        "description": "Build a portfolio with bonds from at least 5 sectors",
        "points": 80
    },
}


# ─────────────────────────────────────────────────────────────
# Quiz Questions
# ─────────────────────────────────────────────────────────────

QUIZ_QUESTIONS = [
    {
        "id": 1,
        "category": "Duration",
        "difficulty": "Easy",
        "question": "What does Modified Duration measure?",
        "options": [
            "The time until a bond matures",
            "The percentage price change for a 1% change in yield",
            "The coupon rate of a bond",
            "The credit risk of a bond"
        ],
        "correct": 1,
        "explanation": "Modified Duration measures the approximate percentage change in bond price for a 1% (100bp) change in yield. It is Macaulay Duration divided by (1 + y/m)."
    },
    {
        "id": 2,
        "category": "Convexity",
        "difficulty": "Medium",
        "question": "Why does positive convexity benefit bondholders?",
        "options": [
            "It increases coupon payments",
            "It makes the bond price rise more when yields fall than it falls when yields rise",
            "It reduces credit risk",
            "It shortens the maturity date"
        ],
        "correct": 1,
        "explanation": "Positive convexity means the price-yield relationship is curved upward. For equal magnitude yield changes, the price gain from a yield decrease exceeds the price loss from a yield increase."
    },
    {
        "id": 3,
        "category": "DV01",
        "difficulty": "Easy",
        "question": "DV01 (Dollar Value of 01) represents:",
        "options": [
            "The bond's dirty price",
            "The price change for a 1 basis point change in yield",
            "The annual coupon payment",
            "The spread over the risk-free rate"
        ],
        "correct": 1,
        "explanation": "DV01 measures the absolute change in bond price for a 1 basis point (0.01%) move in yield. DV01 = Modified Duration × Price × 0.0001."
    },
    {
        "id": 4,
        "category": "Yield Curve",
        "difficulty": "Medium",
        "question": "A 'flattening' yield curve typically indicates:",
        "options": [
            "All yields increasing equally",
            "Short-term rates rising faster than long-term rates, or long-term rates falling faster",
            "Long-term rates rising faster than short-term rates",
            "No change in rates"
        ],
        "correct": 1,
        "explanation": "A flattening curve occurs when the spread between long and short rates narrows. This can happen through bear flattening (short rates rise more) or bull flattening (long rates fall more)."
    },
    {
        "id": 5,
        "category": "Duration",
        "difficulty": "Hard",
        "question": "If a bond has a Modified Duration of 7.5 and Convexity of 85, what is the approximate price change for a +150 bps yield shock?",
        "options": [
            "-11.25% (duration only)",
            "-10.30% (with convexity adjustment)",
            "-7.50%",
            "-15.00%"
        ],
        "correct": 1,
        "explanation": "ΔP/P ≈ -D×Δy + 0.5×C×Δy² = -7.5×0.015 + 0.5×85×0.015² = -0.1125 + 0.009563 = -10.29%. The convexity adjustment reduces the estimated loss."
    },
    {
        "id": 6,
        "category": "Convexity",
        "difficulty": "Hard",
        "question": "Which bond typically has the highest convexity?",
        "options": [
            "Short-maturity, high-coupon bond",
            "Long-maturity, low-coupon (or zero-coupon) bond",
            "Floating-rate note",
            "Callable bond at par"
        ],
        "correct": 1,
        "explanation": "Convexity increases with maturity and decreases with coupon rate. Zero-coupon bonds have the highest convexity for a given maturity because all cashflow is concentrated at maturity."
    },
    {
        "id": 7,
        "category": "Risk",
        "difficulty": "Medium",
        "question": "Value at Risk (VaR) at 95% confidence tells you:",
        "options": [
            "The expected profit over the next day",
            "The maximum loss that will not be exceeded with 95% probability over a given horizon",
            "The average loss across all scenarios",
            "The probability of default"
        ],
        "correct": 1,
        "explanation": "VaR at 95% means there is only a 5% chance the actual loss will exceed the VaR amount. It does NOT represent the maximum possible loss."
    },
    {
        "id": 8,
        "category": "Duration",
        "difficulty": "Easy",
        "question": "Macaulay Duration is best described as:",
        "options": [
            "The weighted average time until a bond's cashflows are received",
            "The bond's modified duration times its yield",
            "The time to the bond's first coupon",
            "The remaining years to maturity"
        ],
        "correct": 0,
        "explanation": "Macaulay Duration is the weighted average time to receive the bond's cashflows, where weights are the present values of each cashflow as a fraction of the bond price."
    },
    {
        "id": 9,
        "category": "Risk",
        "difficulty": "Hard",
        "question": "CVaR (Conditional VaR / Expected Shortfall) differs from VaR because it:",
        "options": [
            "Is always lower than VaR",
            "Measures the average loss in the tail beyond the VaR threshold",
            "Only applies to equity portfolios",
            "Uses parametric distribution only"
        ],
        "correct": 1,
        "explanation": "CVaR (Expected Shortfall) captures the average loss in scenarios where the loss exceeds VaR. It is always ≥ VaR and provides a more complete picture of tail risk."
    },
    {
        "id": 10,
        "category": "Yield Curve",
        "difficulty": "Medium",
        "question": "In yield curve PCA, the first principal component (PC1) typically represents:",
        "options": [
            "The butterfly movement",
            "The slope change",
            "The level (parallel shift) of the yield curve",
            "The twist factor"
        ],
        "correct": 2,
        "explanation": "PC1 explains 85-95% of yield curve variance and represents parallel shifts (level). PC2 captures slope changes (~5-10%), and PC3 captures curvature/butterfly movements (~1-3%)."
    },
]


# ─────────────────────────────────────────────────────────────
# Scenario Challenge Engine
# ─────────────────────────────────────────────────────────────

class ScenarioChallenge:
    """
    Generates interactive yield curve shock scenarios for training.
    Users predict the portfolio impact and earn points.
    """
    
    def __init__(self, bond_df):
        self.bond_df = bond_df
        total_mv = bond_df['MarketValue_INR'].sum()
        weights = bond_df['MarketValue_INR'] / total_mv
        self.port_duration = (weights * bond_df['ModifiedDuration']).sum()
        self.port_convexity = (weights * bond_df['Convexity']).sum()
        self.total_mv = total_mv
    
    def generate_scenario(self, difficulty="Medium"):
        """Generate a random scenario based on difficulty level."""
        np.random.seed(None)  # Truly random
        
        if difficulty == "Easy":
            shock = np.random.choice([-50, -25, 25, 50])
        elif difficulty == "Medium":
            shock = np.random.choice([-100, -75, -50, 50, 75, 100])
        else:  # Hard
            shock = np.random.choice([-200, -150, 150, 200, 250, 300])
        
        dy = shock / 10000.0
        actual_pnl = (-self.port_duration * dy + 0.5 * self.port_convexity * dy**2) * self.total_mv
        actual_pct = actual_pnl / self.total_mv * 100
        
        scenario = {
            'shock_bps': shock,
            'actual_pnl': actual_pnl,
            'actual_pct': actual_pct,
            'duration_effect': -self.port_duration * dy * self.total_mv,
            'convexity_effect': 0.5 * self.port_convexity * dy**2 * self.total_mv,
            'difficulty': difficulty,
        }
        
        return scenario
    
    def evaluate_prediction(self, scenario, user_direction, user_pct_guess=None):
        """
        Evaluate user's prediction against actual outcome.
        
        Parameters
        ----------
        scenario : dict — Generated scenario
        user_direction : str — 'gain' or 'loss'
        user_pct_guess : float — Optional percentage guess
        
        Returns
        -------
        dict — Score and feedback
        """
        actual_direction = 'gain' if scenario['actual_pnl'] > 0 else 'loss'
        direction_correct = (user_direction == actual_direction)
        
        score = 0
        feedback = []
        
        if direction_correct:
            score += 10
            feedback.append("✅ Correct direction!")
        else:
            feedback.append("❌ Wrong direction.")
        
        if user_pct_guess is not None:
            error = abs(user_pct_guess - abs(scenario['actual_pct']))
            if error < 0.5:
                score += 20
                feedback.append(f"🎯 Excellent estimate! (Error: {error:.2f}%)")
            elif error < 1.0:
                score += 10
                feedback.append(f"👍 Good estimate (Error: {error:.2f}%)")
            elif error < 2.0:
                score += 5
                feedback.append(f"🔸 Fair estimate (Error: {error:.2f}%)")
            else:
                feedback.append(f"🔻 Off target (Error: {error:.2f}%)")
        
        return {
            'score': score,
            'direction_correct': direction_correct,
            'feedback': feedback,
            'actual_pnl': scenario['actual_pnl'],
            'actual_pct': scenario['actual_pct'],
        }


# ─────────────────────────────────────────────────────────────
# Trading Simulation
# ─────────────────────────────────────────────────────────────

class TradingSimulation:
    """
    Simulated bond trading to optimize portfolio duration.
    Users buy/sell bonds to hit a target duration.
    """
    
    def __init__(self, bond_df, target_duration=5.0, budget_inr=1e8):
        self.available_bonds = bond_df.copy()
        self.target_duration = target_duration
        self.budget = budget_inr
        self.portfolio = pd.DataFrame()
        self.trades = []
        self.cash = budget_inr
    
    def get_available_bonds(self, n=20):
        """Get a random selection of available bonds."""
        return self.available_bonds.sample(n=min(n, len(self.available_bonds)))
    
    def execute_trade(self, bond_id, quantity):
        """
        Execute a buy/sell trade.
        
        Parameters
        ----------
        bond_id : str — Bond identifier
        quantity : int — Positive for buy, negative for sell
        """
        bond = self.available_bonds[self.available_bonds['BondID'] == bond_id]
        if bond.empty:
            return {"success": False, "message": f"Bond {bond_id} not found"}
        
        bond = bond.iloc[0]
        trade_value = abs(quantity) * bond['DirtyPrice']
        
        if quantity > 0 and trade_value > self.cash:
            return {"success": False, "message": "Insufficient cash"}
        
        trade = {
            'BondID': bond_id,
            'Quantity': quantity,
            'Price': bond['DirtyPrice'],
            'Duration': bond['ModifiedDuration'],
            'Convexity': bond['Convexity'],
            'Value': trade_value,
            'Timestamp': datetime.now().isoformat(),
        }
        
        self.trades.append(trade)
        
        if quantity > 0:
            self.cash -= trade_value
            # Add to portfolio
            new_row = bond.to_frame().T.copy()
            new_row['Quantity'] = quantity
            new_row['MarketValue_INR'] = trade_value
            self.portfolio = pd.concat([self.portfolio, new_row], ignore_index=True)
        
        return {"success": True, "message": f"Executed: {'Buy' if quantity > 0 else 'Sell'} {abs(quantity)} × {bond_id}", "trade": trade}
    
    def portfolio_metrics(self):
        """Calculate current portfolio metrics."""
        if self.portfolio.empty:
            return {
                'duration': 0, 'convexity': 0, 'total_value': 0,
                'cash': self.cash, 'n_bonds': 0,
                'duration_gap': self.target_duration, 'score': 0
            }
        
        total_mv = self.portfolio['MarketValue_INR'].sum()
        weights = self.portfolio['MarketValue_INR'] / total_mv
        
        port_dur = (weights * self.portfolio['ModifiedDuration']).sum()
        port_conv = (weights * self.portfolio['Convexity']).sum()
        dur_gap = abs(port_dur - self.target_duration)
        
        # Score: higher is better (max 100)
        score = max(0, 100 - dur_gap * 20)
        
        return {
            'duration': port_dur,
            'convexity': port_conv,
            'total_value': total_mv,
            'cash': self.cash,
            'n_bonds': len(self.portfolio),
            'duration_gap': dur_gap,
            'score': score,
        }


# ─────────────────────────────────────────────────────────────
# Risk Profiling
# ─────────────────────────────────────────────────────────────

RISK_PROFILES = {
    "Conservative": {
        "target_duration": 3.0,
        "max_convexity": 30,
        "description": "Low duration, focus on capital preservation",
        "sectors": ["Government"],
        "icon": "🛡️"
    },
    "Moderate": {
        "target_duration": 5.5,
        "max_convexity": 80,
        "description": "Balanced duration, mix of government and corporate",
        "sectors": ["Government", "Financial"],
        "icon": "⚖️"
    },
    "Aggressive": {
        "target_duration": 9.0,
        "max_convexity": 200,
        "description": "Long duration, higher yield potential",
        "sectors": ["Government", "Financial", "Corporate"],
        "icon": "🚀"
    },
    "Barbell": {
        "target_duration": 6.0,
        "max_convexity": 150,
        "description": "Mix of short and long duration, high convexity",
        "sectors": ["Government", "Corporate"],
        "icon": "🏋️"
    },
}


# ─────────────────────────────────────────────────────────────
# Game Session Manager
# ─────────────────────────────────────────────────────────────

class GameSession:
    """Manages a gamified training session with score tracking."""
    
    def __init__(self, player_name="Analyst"):
        self.player_name = player_name
        self.score = 0
        self.achievements = []
        self.scenario_history = []
        self.quiz_results = []
        self.start_time = datetime.now()
    
    def add_score(self, points):
        self.score += points
    
    def unlock_achievement(self, achievement_id):
        if achievement_id in ACHIEVEMENTS and achievement_id not in self.achievements:
            self.achievements.append(achievement_id)
            ach = ACHIEVEMENTS[achievement_id]
            self.score += ach['points']
            return True
        return False
    
    def run_quiz(self, questions=None):
        """Run quiz and return results (for console/API mode)."""
        if questions is None:
            questions = QUIZ_QUESTIONS
        
        results = []
        correct_count = 0
        
        for q in questions:
            results.append({
                'question_id': q['id'],
                'category': q['category'],
                'difficulty': q['difficulty'],
                'question': q['question'],
                'correct_answer': q['options'][q['correct']],
                'explanation': q['explanation'],
            })
        
        return results
    
    def get_summary(self):
        """Get session summary."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'player': self.player_name,
            'score': self.score,
            'achievements': [ACHIEVEMENTS[a] for a in self.achievements if a in ACHIEVEMENTS],
            'scenarios_completed': len(self.scenario_history),
            'quiz_questions': len(self.quiz_results),
            'elapsed_seconds': elapsed,
            'rank': self._calculate_rank(),
        }
    
    def _calculate_rank(self):
        if self.score >= 800:
            return "🏅 Senior Portfolio Manager"
        elif self.score >= 500:
            return "📈 Fixed Income Analyst"
        elif self.score >= 300:
            return "📊 Junior Analyst"
        elif self.score >= 100:
            return "📚 Trainee"
        else:
            return "🌱 Beginner"


# ─────────────────────────────────────────────────────────────
# Main Execution (Console Demo)
# ─────────────────────────────────────────────────────────────

def run_part6():
    """Execute Part 6: Bond Risk Lab demo."""
    print_section_header("PART 6: Bond Risk Lab — Gamified Simulation Platform")
    
    # 1. Load data
    print("  📂 Loading bond data...")
    bond_df = load_bond_portfolio()
    
    # 2. Initialize game
    session = GameSession("Demo Analyst")
    challenge = ScenarioChallenge(bond_df)
    
    # 3. Demo: Scenario Challenges
    print_subsection("Scenario Challenge Demo")
    
    for difficulty in ["Easy", "Medium", "Hard"]:
        scenario = challenge.generate_scenario(difficulty)
        direction = 'gain' if scenario['actual_pnl'] > 0 else 'loss'
        result = challenge.evaluate_prediction(scenario, direction, abs(scenario['actual_pct']))
        
        print(f"  [{difficulty}] Shock: {scenario['shock_bps']:+d} bps")
        print(f"     P&L: ₹{scenario['actual_pnl']/1e5:+.2f} Lakhs ({scenario['actual_pct']:+.4f}%)")
        print(f"     Duration Effect: ₹{scenario['duration_effect']/1e5:+.2f} L")
        print(f"     Convexity Effect: ₹{scenario['convexity_effect']/1e5:+.2f} L")
        print(f"     Score: +{result['score']}")
        print()
        
        session.add_score(result['score'])
        session.scenario_history.append(scenario)
    
    session.unlock_achievement('duration_master')
    
    # 4. Demo: Trading Simulation
    print_subsection("Trading Simulation Demo")
    sim = TradingSimulation(bond_df, target_duration=5.0, budget_inr=1e8)
    
    # Auto-select bonds to demonstrate
    sample_bonds = bond_df.nsmallest(5, 'ModifiedDuration')
    for _, bond in sample_bonds.iterrows():
        result = sim.execute_trade(bond['BondID'], 100)
        if result['success']:
            print(f"  {result['message']}")
    
    metrics = sim.portfolio_metrics()
    print(f"\n  Portfolio Duration: {metrics['duration']:.2f} (Target: {sim.target_duration})")
    print(f"  Duration Gap: {metrics['duration_gap']:.2f}")
    print(f"  Portfolio Score: {metrics['score']:.0f}/100")
    print(f"  Bonds: {metrics['n_bonds']}, Cash: ₹{metrics['cash']/1e7:.2f} Cr")
    
    if metrics['duration_gap'] < 0.5:
        session.unlock_achievement('risk_manager')
    
    # 5. Demo: Quiz
    print_subsection("Quiz Sample")
    quiz_results = session.run_quiz(QUIZ_QUESTIONS[:3])
    for r in quiz_results:
        print(f"  Q{r['question_id']} [{r['difficulty']}]: {r['question']}")
        print(f"     Answer: {r['correct_answer']}")
        print(f"     {r['explanation'][:80]}...")
        print()
    
    session.unlock_achievement('perfect_score')
    
    # 6. Risk Profiles
    print_subsection("Risk Profiles")
    for name, profile in RISK_PROFILES.items():
        print(f"  {profile['icon']} {name}: Duration Target={profile['target_duration']}, "
              f"Sectors={profile['sectors']}")
    
    # 7. Achievements
    print_subsection("Unlocked Achievements")
    for ach_id in session.achievements:
        ach = ACHIEVEMENTS[ach_id]
        print(f"  {ach['icon']} {ach['name']} (+{ach['points']} pts) — {ach['description']}")
    
    # 8. Session Summary
    print_subsection("Session Summary")
    summary = session.get_summary()
    print(f"  Player: {summary['player']}")
    print(f"  Score: {summary['score']}")
    print(f"  Rank: {summary['rank']}")
    print(f"  Scenarios: {summary['scenarios_completed']}")
    print(f"  Achievements: {len(summary['achievements'])}")
    
    # 9. Save session data
    session_data = {
        'quiz_questions': QUIZ_QUESTIONS,
        'achievements': ACHIEVEMENTS,
        'risk_profiles': RISK_PROFILES,
    }
    with open(REPORTS_DIR / "part6_game_config.json", 'w') as f:
        json.dump(session_data, f, indent=2)
    
    print_section_header("PART 6 COMPLETE ✓")
    return session


if __name__ == "__main__":
    run_part6()
