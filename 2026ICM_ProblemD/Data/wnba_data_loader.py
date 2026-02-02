"""
WNBA ICM 2026 Data Loader
=========================
Helper functions to load and process WNBA data for ICM Problem D

Usage:
    from wnba_data_loader import WNBADataLoader
    loader = WNBADataLoader('./wnba_data/')
    teams = loader.load_team_valuations()
    params = loader.get_model_parameters()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


class WNBADataLoader:
    """Load and process WNBA data for ICM modeling"""
    
    def __init__(self, data_dir: str = './'):
        self.data_dir = Path(data_dir)
        self._cache = {}
    
    def load_team_valuations(self) -> pd.DataFrame:
        """Load team valuations data"""
        if 'valuations' not in self._cache:
            self._cache['valuations'] = pd.read_csv(
                self.data_dir / 'team_valuations.csv'
            )
        return self._cache['valuations']
    
    def load_salary_cap_history(self) -> pd.DataFrame:
        """Load historical salary cap data"""
        if 'salary_cap' not in self._cache:
            self._cache['salary_cap'] = pd.read_csv(
                self.data_dir / 'salary_cap_history.csv'
            )
        return self._cache['salary_cap']
    
    def load_player_salaries(self) -> pd.DataFrame:
        """Load 2025 player salary data"""
        if 'salaries' not in self._cache:
            self._cache['salaries'] = pd.read_csv(
                self.data_dir / 'player_salaries_2025.csv'
            )
        return self._cache['salaries']
    
    def load_expansion_teams(self) -> pd.DataFrame:
        """Load expansion team data"""
        if 'expansion' not in self._cache:
            self._cache['expansion'] = pd.read_csv(
                self.data_dir / 'expansion_teams.csv'
            )
        return self._cache['expansion']
    
    def load_attendance(self) -> pd.DataFrame:
        """Load attendance data"""
        if 'attendance' not in self._cache:
            self._cache['attendance'] = pd.read_csv(
                self.data_dir / 'attendance_data.csv'
            )
        return self._cache['attendance']
    
    def load_media_deals(self) -> pd.DataFrame:
        """Load media deals data"""
        if 'media' not in self._cache:
            self._cache['media'] = pd.read_csv(
                self.data_dir / 'media_deals.csv'
            )
        return self._cache['media']
    
    def load_model_parameters(self) -> pd.DataFrame:
        """Load model parameters"""
        if 'params' not in self._cache:
            self._cache['params'] = pd.read_csv(
                self.data_dir / 'model_parameters.csv'
            )
        return self._cache['params']
    
    def get_parameter(self, param_name: str) -> Dict[str, Any]:
        """Get a specific parameter with all its values"""
        params = self.load_model_parameters()
        row = params[params['parameter'] == param_name]
        if row.empty:
            # Try by symbol
            row = params[params['symbol'] == param_name]
        if row.empty:
            raise ValueError(f"Parameter '{param_name}' not found")
        return row.iloc[0].to_dict()
    
    def get_model_parameters_dict(self) -> Dict[str, float]:
        """Get all parameters as a dictionary {symbol: suggested_value}"""
        params = self.load_model_parameters()
        return dict(zip(params['symbol'], params['suggested_value']))
    
    def get_team_data(self, team_name: str) -> Dict[str, Any]:
        """Get comprehensive data for a specific team"""
        valuations = self.load_team_valuations()
        attendance = self.load_attendance()
        
        team_val = valuations[valuations['team'].str.contains(team_name, case=False)]
        team_att = attendance[attendance['team'].str.contains(team_name, case=False)]
        
        if team_val.empty:
            raise ValueError(f"Team '{team_name}' not found")
        
        result = team_val.iloc[0].to_dict()
        if not team_att.empty:
            result.update({
                f'attendance_{k}': v 
                for k, v in team_att.iloc[0].to_dict().items() 
                if k != 'team'
            })
        return result


# ============= Model Helper Functions =============

def calculate_player_value(
    performance_stats: Dict[str, float],
    commercial_metrics: Dict[str, float],
    age: int,
    alpha: float = 0.7,
    peak_age: int = 27,
    decay_rate: float = 0.02
) -> float:
    """
    Calculate total player value combining performance and commercial value
    
    V_total = alpha * V_perf * age_factor + (1-alpha) * V_comm
    
    Parameters:
    -----------
    performance_stats : dict with keys like 'ppg', 'rpg', 'apg', 'per'
    commercial_metrics : dict with keys like 'social_followers', 'endorsement_value'
    age : player's age
    alpha : weight on performance (vs commercial)
    peak_age : age of peak performance
    decay_rate : rate of decay from peak (kappa)
    
    Returns:
    --------
    float : total player value (normalized)
    """
    # Performance value (simplified PER-based)
    per = performance_stats.get('per', 15)
    v_perf = per / 15.0  # Normalize to league average
    
    # Age adjustment (Gaussian decay)
    age_factor = np.exp(-decay_rate * (age - peak_age) ** 2)
    
    # Commercial value (simplified)
    followers = commercial_metrics.get('social_followers', 0)
    endorsements = commercial_metrics.get('endorsement_value', 0)
    all_star = commercial_metrics.get('all_star_selections', 0)
    
    v_comm = (
        0.3 * np.log1p(followers) / np.log1p(2_000_000) +  # Social media
        0.4 * endorsements / 5_000_000 +  # Endorsement value (normalized to $5M)
        0.3 * min(all_star, 5) / 5  # All-star selections (cap at 5)
    )
    
    return alpha * v_perf * age_factor + (1 - alpha) * v_comm


def estimate_ticket_demand(
    base_price: float,
    price: float,
    elasticity: float = -1.0,
    opponent_rank: int = 8,
    is_weekend: bool = False,
    is_rivalry: bool = False,
    has_promo: bool = False
) -> float:
    """
    Estimate ticket demand using log-linear demand function
    
    ln(ATT) = β₀ + β₁*OppRank + β₂*Weekend + β₃*Rivalry + β₄*Promo + η*ln(Price)
    
    Parameters:
    -----------
    base_price : baseline average ticket price
    price : actual ticket price
    elasticity : price elasticity (η, typically negative)
    opponent_rank : opponent's league rank (1=best)
    is_weekend : whether game is on weekend
    is_rivalry : whether it's a rivalry game
    has_promo : whether there's a promotion
    
    Returns:
    --------
    float : demand multiplier (1.0 = baseline demand)
    """
    # Coefficients (estimated)
    beta_opp = -0.02  # Better opponents draw more
    beta_weekend = 0.15  # Weekend boost
    beta_rivalry = 0.25  # Rivalry boost
    beta_promo = 0.10  # Promotion boost
    
    # Calculate demand adjustment
    ln_mult = (
        beta_opp * (8 - opponent_rank) +  # Relative to middle team
        beta_weekend * int(is_weekend) +
        beta_rivalry * int(is_rivalry) +
        beta_promo * int(has_promo) +
        elasticity * np.log(price / base_price)
    )
    
    return np.exp(ln_mult)


def calculate_win_rate_pythagorean(
    points_for: float,
    points_against: float,
    exponent: float = 13.91
) -> float:
    """
    Calculate expected win rate using Pythagorean expectation
    
    WP = PF^exp / (PF^exp + PA^exp)
    
    Parameters:
    -----------
    points_for : average points scored per game
    points_against : average points allowed per game
    exponent : Pythagorean exponent (NBA/WNBA typical ~13.91)
    
    Returns:
    --------
    float : expected win percentage (0-1)
    """
    pf_exp = points_for ** exponent
    pa_exp = points_against ** exponent
    return pf_exp / (pf_exp + pa_exp)


def project_salary_cap(
    current_year: int,
    current_cap: float,
    target_year: int,
    growth_rate: float = 0.03,
    new_cba_bump: Optional[float] = None,
    new_cba_year: int = 2026
) -> float:
    """
    Project future salary cap
    
    Parameters:
    -----------
    current_year : starting year
    current_cap : current salary cap
    target_year : year to project to
    growth_rate : annual growth rate (default 3% under current CBA)
    new_cba_bump : one-time increase when new CBA takes effect
    new_cba_year : year new CBA takes effect
    
    Returns:
    --------
    float : projected salary cap
    """
    years = target_year - current_year
    projected = current_cap * (1 + growth_rate) ** years
    
    if new_cba_bump and target_year >= new_cba_year:
        projected *= (1 + new_cba_bump)
    
    return projected


# ============= Example Usage =============

if __name__ == '__main__':
    # Initialize loader
    loader = WNBADataLoader('./')
    
    # Load data
    print("=== Team Valuations ===")
    teams = loader.load_team_valuations()
    print(teams[['team', 'valuation_2024_M', 'valuation_2025_M']].head())
    
    print("\n=== Model Parameters ===")
    params = loader.get_model_parameters_dict()
    print(f"Salary Cap 2025: ${params['S_cap']:,.0f}")
    print(f"Price Elasticity: {params['eta']}")
    print(f"Discount Rate: {params['r']:.1%}")
    
    print("\n=== Las Vegas Aces Data ===")
    aces = loader.get_team_data('Las Vegas')
    print(f"2025 Valuation: ${aces['valuation_2025_M']}M")
    print(f"2024 Revenue: ${aces['revenue_2024_M']}M")
    
    print("\n=== A'ja Wilson Value Estimate ===")
    wilson_value = calculate_player_value(
        performance_stats={'per': 36.5, 'ppg': 26.9, 'rpg': 11.9},
        commercial_metrics={
            'social_followers': 2_000_000,
            'endorsement_value': 5_000_000,
            'all_star_selections': 6
        },
        age=28
    )
    print(f"Normalized Value Score: {wilson_value:.3f}")
    
    print("\n=== Demand Estimation Example ===")
    demand = estimate_ticket_demand(
        base_price=80,
        price=100,
        elasticity=-1.0,
        opponent_rank=1,  # Playing top team
        is_weekend=True,
        is_rivalry=True
    )
    print(f"Demand Multiplier: {demand:.2f}x baseline")
    
    print("\n=== Win Rate Projection ===")
    wp = calculate_win_rate_pythagorean(85, 78)
    print(f"Expected Win Rate (85 PPG for, 78 against): {wp:.1%}")
