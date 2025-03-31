🔍 Overview
This project analyzes annual banking data to uncover customer behaviour, segmentation, spending patterns, and key financial insights using data visualization and machine learning techniques.

📂 Dataset
The dataset contains financial information, including account balances, income, credit scores, and transaction behavior.
Missing values are handled using mean imputation.
Features are standardized for clustering analysis.

📈 Analysis & Methodology
1️⃣ Data Preprocessing:
Cleaned column names and handled missing values.
Standardized numerical features using StandardScaler().

2️⃣ Exploratory Data Analysis (EDA)
Histograms & KDE Plots: Visualizing the distribution of account balance and income.
Pie Charts: Showing categorical distributions (Gender, Location, Account Type).
Violin Plots: Understanding the spread of account balance and income.
Heatmap: Identifying correlations between numerical features.

3️⃣ Customer Segmentation using K-Means Clustering.
Applied K-Means clustering with an optimal k=4 determined using the elbow method.
Visualized clusters in a scatter plot to understand customer groups.
Created radar plots for cluster profile comparison.

📊 Visualizations
The project includes:
✔️ Histograms of financial attributes
✔️ Pie charts for categorical distributions
✔️ Correlation heatmaps
✔️ Violin plots for spread analysis
✔️ K-Means clustering with customer segmentation

🛠️ Technologies Used
Python : Coding,
Pandas: Data manipulation,
Numpy : To push numerical Values,
Matplotlib & Seaborn: Data visualization and
Scikit-learn: Machine learning (K-Means clustering).

🚀 How to Use
1️⃣ Clone this repository:
git clone https://github.com/your-repo/bank-annual-analysis.git
2️⃣ Install dependencies:
pip install pandas matplotlib seaborn scikit-learn numpy
3️⃣ Run the analysis script:
python analysis.py
📌 Conclusion
The analysis provides valuable insights into customer financial behavior.
Customer segmentation can help banks tailor their services.
Further improvements could include advanced clustering techniques or predictive modeling.

