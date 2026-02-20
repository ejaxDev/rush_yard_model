# rush_yard_model
# 🏈 NFL Rush Yards Prediction Model

An end-to-end machine learning pipeline that predicts NFL running back rushing yards per game using XGBoost regression with extensive feature engineering and opponent-adjusted metrics.

## 🎯 Project Overview

This project builds a sophisticated predictive model for NFL player rushing performance by engineering over 100+ features across multiple dimensions: player history, team dynamics, opponent strength, play-by-play patterns, and advanced metrics like yards before/after contact. The model employs walk-forward validation across multiple NFL seasons (2018-2025) to ensure robust, leak-free predictions.

### Key Features
- **Multi-dimensional feature engineering** across player, team, and opponent perspectives
- **Leak-free rolling statistics** using lag-shifted windows (1, 3, 5, 10 games)
- **Opponent-adjusted defensive metrics** accounting for offensive strength
- **Advanced rushing analytics** including YBC/YAC, zone efficiency, and carry distribution
- **Walk-forward validation** for realistic time-series evaluation
- **Permutation-based feature selection** to identify predictive drivers

## 📊 Data Sources

The model ingests NFL rushing data from AWS S3 across multiple tables:
- `base_stats` – Game-level rushing statistics (yards, attempts, touchdowns, success rate)
- `play_by_play` – Individual play-level data with zone information
- `ybc_yac` – Yards before contact / yards after contact metrics
- `defense_stats` – Defensive performance against the run

**Data Range:** 2018-2025 NFL seasons

## 🏗️ Architecture

### Feature Engineering Pipeline

The project is organized into modular Jupyter notebooks, each handling a specific feature domain:

#### 1. **Base Stat Feature Engineering** (`base_stat_feature_engineering.ipynb`)
Creates foundational player rolling statistics:
- Rolling averages (1, 3, 5, 10 games): yards, attempts, YPC, success rate
- Momentum indicators: delta between short/long windows
- Consistency metrics: volatility (std) of performance
- Workload distribution: carry share, teammate competition, injury impact

**Output:** `base_stats_feature_engineering.csv`

#### 2. **Defense Feature Engineering** (`defense_feature_engineering.ipynb`)
Builds opponent-adjusted defensive metrics:
- Yards/YPC allowed by defense (rolling windows)
- Strength-of-offense adjustments for matchup quality
- Defensive performance relative to expected (opponent-adjusted)
- Momentum deltas and volatility measures

**Output:** `defense_stats_feature_engineering.csv`

#### 3. **Play-by-Play Feature Engineering** (`play_by_play_feature_engineering.ipynb`)
Extracts granular efficiency metrics from individual plays:
- Carry distribution buckets (≤0, 1-2, 3-5, 6+, 10+, 20+, 40+ yards)
- Zone-level efficiency (left/middle/right differentials)
- Features computed at player, team, and opponent levels

**Output:** `play_by_play_feature_engineering.csv`

#### 4. **YBC/YAC Feature Engineering** (`ybc_yac_feature_engineering.ipynb`)
Analyzes contact-based rushing metrics:
- Yards before contact (YBC) – offensive line quality indicator
- Yards after contact (YAC) – running back elusiveness
- Broken tackles per attempt
- Rolling trends at player, team, and opponent perspectives

**Output:** `ybc_yac_feature_engineering.csv`

### Model Training & Evaluation

#### **Regression Model** (`train_test_reg.ipynb`)
- **Algorithm:** XGBoost Regressor
- **Target:** Rushing yards per game
- **Validation:** Walk-forward by season (train on past, test on future)
- **Feature Selection:** Permutation importance with iterative pruning
- **Evaluation Metrics:** MAE, RMSE, R²

**Pipeline:**
1. Load and merge all engineered feature CSVs
2. Normalize player names for consistent joins
3. Define feature set and examine correlations
4. Walk-forward training loop (2019-2023 test seasons)
5. Permutation importance analysis
6. Retrain with refined feature set
7. Export model to JSON

## 🔑 Key Technical Highlights

### Data Leakage Prevention
All rolling statistics use `.shift(1)` to ensure features for game *N* only reflect data from games *N-1, N-2, ...*. This prevents the model from "seeing the future" and ensures realistic backtesting.

### Opponent Adjustment
Defensive metrics are adjusted for the quality of opposing offenses faced, providing a more accurate assessment of true defensive strength rather than raw statistics.

### Feature Diversity
The model combines:
- **Historical performance** (rolling averages)
- **Momentum signals** (short vs. long-term trends)
- **Matchup context** (opponent defense quality)
- **Situation-specific patterns** (carry distribution, zone efficiency)
- **Advanced metrics** (YBC, YAC, broken tackles)

### Modular Design
Each feature engineering notebook is self-contained, outputs a CSV, and can be run independently. The training notebook simply merges these outputs, enabling:
- Easy experimentation with new feature groups
- Clear separation of concerns
- Reproducible pipeline execution

## 📁 Project Structure

```
rush_yard_model/
├── feature_engineering/
│   ├── base_stat_feature_engineering.ipynb
│   ├── defense_feature_engineering.ipynb
│   ├── play_by_play_feature_engineering.ipynb
│   └── ybc_yac_feature_engineering.ipynb
├── train_test/
│   └── train_test_reg.ipynb
├── .gitignore
└── README.md
```

## 🚀 Usage

### Prerequisites
- Python 3.8+
- pandas, numpy, xgboost, scikit-learn
- AWS credentials configured for S3 access
- Custom `utils` module for S3 data loading

### Running the Pipeline

1. **Feature Engineering** (run in order or parallel):
```bash
jupyter notebook feature_engineering/base_stat_feature_engineering.ipynb
jupyter notebook feature_engineering/defense_feature_engineering.ipynb
jupyter notebook feature_engineering/play_by_play_feature_engineering.ipynb
jupyter notebook feature_engineering/ybc_yac_feature_engineering.ipynb
```

2. **Model Training**:
```bash
jupyter notebook train_test/train_test_reg.ipynb
```

Each notebook outputs a CSV to disk that subsequent notebooks can consume.

## 📈 Results & Performance

The model is evaluated using walk-forward validation across multiple NFL seasons. Performance metrics (MAE, RMSE, R²) are computed for each test season and averaged to assess generalization to unseen data.

**Key Insights:**
- Permutation importance reveals which feature groups (player history, defense matchup, play-by-play patterns) drive predictions
- Momentum features (deltas) capture form trends beyond simple averages
- Opponent-adjusted metrics significantly improve predictive accuracy

## 🛠️ Technologies Used

- **Python** – Primary programming language
- **Pandas** – Data manipulation and feature engineering
- **NumPy** – Numerical computations
- **XGBoost** – Gradient boosting regression model
- **Scikit-learn** – Model evaluation and feature selection
- **Jupyter Notebook** – Interactive development and documentation
- **AWS S3** – Cloud data storage and retrieval

## 📝 Future Enhancements

- [ ] Incorporate weather data (temperature, wind, precipitation)
- [ ] Add Vegas betting lines as market-implied expectations
- [ ] Build classification models for over/under betting markets
- [ ] Real-time prediction API for in-season usage
- [ ] Feature importance visualization dashboard
- [ ] Expand to receiving yards and touchdown predictions

## 👤 Author

**ejaxDev**  
[GitHub](https://github.com/ejaxDev) | [Repository](https://github.com/ejaxDev/rush_yard_model)

## 📄 License

This project is available for portfolio and educational purposes.

---

*This model was built for research and portfolio purposes. NFL data sourced from publicly available statistics.*
