# 🧮 Mathematical Model for Production Plan Optimization

## 📘 Overview

This project focuses on developing **mathematical optimization models** to improve production planning and decision-making.  
It includes a **base model**, an **extended model**, and a **sensitivity analysis module**, allowing users to test how different parameters affect optimal production outcomes.

The models are built in **Python** using optimization techniques to minimize production costs and efficiently allocate limited resources while meeting demand.

---

## 🎯 Objectives

- Formulate a **production optimization model** under demand and capacity constraints.  
- Extend the model with additional factors for real-world applicability.  
- Perform **sensitivity analysis** to assess model robustness.  
- Support **data-driven decision making** for production managers.

---

## 🧩 Project Structure

```
.
├── Mathematical Model Codebase/        # Main Python codebase
│   ├── base_model.py                   # Base linear optimization model
│   ├── extended_model.py               # Extended model with extra constraints
│   ├── sensitivity_analysis.py         # Sensitivity analysis script
│   ├── run_models.py                   # Script to run and compare models
│   └── utils/                          # Helper scripts (if applicable)
│
├── base_and_extended_model_output/     # Model outputs (tables, results, graphs)
│   ├── base_model_results.csv
│   ├── extended_model_results.csv
│   └── comparison_summary.txt
│
├── sensitivity_analysis_output/        # Sensitivity test results
│   ├── sensitivity_results.csv
│   └── charts/
│
├── article.pdf                         # Full report or academic documentation
└── README.md                           # Project documentation
```

---

## 🛠️ Technologies Used

- **Language:** Python 3.8+  
- **Optimization Library:** PuLP / Pyomo (adjust if different)  
- **Data Handling:** pandas, numpy  
- **Visualization:** matplotlib  
- **Documentation:** article.pdf  

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yashpatel1203/Mathematical-Model-for-Production-Plan-Optimization-.git
   cd Mathematical-Model-for-Production-Plan-Optimization-
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Mac/Linux
   venv\Scripts\activate      # For Windows
   ```

3. **Install required libraries:**
   ```bash
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` isn’t provided, manually install:)*  
   ```bash
   pip install numpy pandas matplotlib pulp pyomo
   ```

---

## ▶️ Running the Models

### Run Base Model
```bash
python "Mathematical Model Codebase/base_model.py"
```

### Run Extended Model
```bash
python "Mathematical Model Codebase/extended_model.py"
```

### Compare Both Models
```bash
python "Mathematical Model Codebase/run_models.py"
```

### Perform Sensitivity Analysis
```bash
python "Mathematical Model Codebase/sensitivity_analysis.py"
```

All results will be saved inside the respective output folders:
- `base_and_extended_model_output/`
- `sensitivity_analysis_output/`

---

## 📊 Example Outputs

| Model Type | Objective Value | Key Insights |
|-------------|-----------------|---------------|
| Base Model | Lower cost | Simpler constraints, fewer resource limits |
| Extended Model | Slightly higher cost | Includes realistic constraints (e.g. multi-product, limited labor) |
| Sensitivity Analysis | Varies | Tests robustness under demand/resource changes |

You can visualize trends or parameter effects using the generated graphs in `charts/`.

---

## 📖 Model Description

**Base Model:**
- Objective: Minimize total production cost  
- Constraints: Demand fulfillment, resource capacity, and non-negativity  

**Extended Model:**
- Adds complexity such as multiple time periods, cost variations, or production switching costs.  
- Reflects real-world production environments more accurately.  

**Sensitivity Analysis:**
- Tests model robustness by altering demand, costs, or capacity.  
- Helps understand which variables most affect the optimal plan.

---

## 📈 Insights

- Mathematical optimization provides quantifiable, reproducible plans.  
- Sensitivity analysis identifies risk areas and trade-offs.  
- The extended model improves decision accuracy but requires more computational effort.  

---

## 🚀 Future Enhancements

- Integrate real industrial datasets.  
- Develop a GUI or web dashboard for running models visually.  
- Add stochastic (uncertainty-based) optimization.  
- Automate reporting and visualization of model performance.  

---

## 👨‍💻 Author

**Yash Patel**  
📧 *Original Project Author*  
🔗 [GitHub Profile](https://github.com/yashpatel1203)

---

## 📜 License

This project is provided for academic and educational use.  
Users are free to reference, modify, and extend the work with proper credit.

---

## 🏷️ Citation

If you use this project for academic or research purposes, please cite:

> Patel, Yash. “Mathematical Model for Production Plan Optimization.” GitHub, 2024.  
> [https://github.com/yashpatel1203/Mathematical-Model-for-Production-Plan-Optimization-](https://github.com/yashpatel1203/Mathematical-Model-for-Production-Plan-Optimization-)
