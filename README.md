🤖 Artificial Intelligence & Machine Learning Essentials

A Short Course on the Mathematical Foundations and Computational Applications of AI and ML

Instructor: Americo Cunha Jr.
Institutions: Laboratório Nacional de Computação Científica (LNCC) and Universidade do Estado do Rio de Janeiro (UERJ), Brazil.

⸻

📖 Overview

Welcome to Artificial Intelligence & Machine Learning Essentials, a short course introducing fundamental concepts, mathematical principles, and computational techniques in artificial intelligence (AI) and machine learning (ML).

The course emphasizes the connections between machine learning, numerical linear algebra, optimization, statistical modeling, and nonlinear dynamical systems.

Rather than treating learning algorithms as black boxes, the computational examples illustrate how mathematical models, numerical methods, and data interact to produce predictions, classifications, and reduced-order representations.

The repository provides practical examples originally implemented in MATLAB, together with corresponding Python implementations based on open-source scientific computing libraries.

The material is intended for educational purposes and can also serve as a starting point for independent computational experiments.

⸻

🎯 Learning Objectives

By the end of the course, participants should be able to:

* 📈 Understand and implement linear, nonlinear, and robust regression techniques.
* 🎯 Apply supervised learning algorithms to classification problems.
* 🧩 Explore unsupervised learning through clustering methods.
* 🧮 Understand singular value decomposition and its role in dimensionality reduction.
* 🧠 Construct elementary neural-network models for nonlinear systems.
* 🌪️ Investigate machine learning applications to chaotic dynamical systems.
* 🔬 Evaluate model predictions and understand numerical and statistical limitations.
* 💻 Implement and visualize learning algorithms using MATLAB or Python.

⸻

📚 Course Topics

📈 1. Regression and Parameter Estimation

Regression provides a mathematical framework for relating observations to explanatory variables and estimating unknown model parameters.

Computational examples:

Example	Description
🪂 Free Fall	Quadratic least-squares fitting of motion data
☄️ Asteroid Trajectory	Implicit conic fitting from trajectory observations
📉 Robust Regression	Comparison of L1 and L2 regression
🦠 COVID-19 Data	Nonlinear fitting of a logistic growth model

Main concepts: least squares, optimization, parameter identification, nonlinear regression, and robustness to outliers.

🎯 2. Supervised Classification

Classification aims to assign observations to categories based on labeled training data.

Computational examples:

Example	Description
📍 k-Nearest Neighbors (k-NN)	Distance-based classification
⚖️ Support Vector Machines (SVM)	Margin-based classification and decision boundaries

Main concepts: feature spaces, classification boundaries, model complexity, and generalization.

🧩 3. Unsupervised Learning

Unsupervised learning explores patterns and structures in datasets without predefined labels.

Computational examples:

Example	Description
🔵 k-Means	Partitioning observations into clusters
🌳 Hierarchical Clustering	Dendrogram construction and hierarchical grouping

Main concepts: similarity measures, clustering objectives, partitioning, and hierarchical organization.

🧮 4. Dimensionality Reduction

High-dimensional information can often be approximated using a small number of dominant components.

Computational example:

Example	Description
🖼️ Image Compression	Low-rank approximation using singular value decomposition (SVD)

Main concepts: matrix factorization, singular values, low-rank approximation, information compression, and reconstruction error.

🧠 5. Artificial Neural Networks

Neural networks approximate nonlinear mappings from data and can be used to construct surrogate models for complex physical systems.

Computational example:

Example	Description
🦋 Lorenz System	Neural-network surrogate modeling of nonlinear dynamics

Main concepts: nonlinear approximation, training, prediction, generalization, and dynamical sensitivity.

🌪️ 6. Reservoir Computing and Chaotic Forecasting

Reservoir computing employs a recurrent dynamical system to transform input signals into a higher-dimensional representation, while training a comparatively simple readout.

Computational example:

Example	Description
🔮 Rössler System	Chaotic time-series forecasting using an Echo State Network (ESN)

Main concepts: recurrent networks, reservoir dynamics, regularized regression, autonomous prediction, and forecast horizons.

⸻

📂 Repository Structure

The recommended repository organization separates MATLAB and Python implementations while retaining the original datasets.

ShortCourse-AI-ML-Essentials/
│
├── README.md
│
├── Main*.m
├── Image2Matrix.m
├── Matrix2Image.m
├── COVID19RJ.mat
├── LaTourEiffel.jpg
│
└── python/
    ├── README.md
    ├── requirements.txt
    │
    ├── Main*.py
    ├── Image2Matrix.py
    ├── Matrix2Image.py
    │
    ├── COVID19RJ.mat
    ├── LaTourEiffel.jpg
    │
    └── tests/

📝 Note: The python/ directory represents the proposed organization for integrating the Python implementations. The original MATLAB examples can be executed independently.

⸻

🗂️ Computational Examples

#	Topic	MATLAB	Python
01	🪂 Free-Fall Regression	MainRegressionFreeFall.m	MainRegressionFreeFall.py
02	☄️ Asteroid Regression	MainRegressionAsteroid.m	MainRegressionAsteroid.py
03	📉 Robust Regression	MainRegressionRobust.m	MainRegressionRobust.py
04	🦠 COVID-19 Regression	MainRegressionCOVID19.m	MainRegressionCOVID19.py
05	📍 k-NN Classification	MainClassificationKNN.m	MainClassificationKNN.py
06	⚖️ SVM Classification	MainClassificationSVM.m	MainClassificationSVM.py
07	🔵 k-Means Clustering	MainClusteringKmeans.m	MainClusteringKmeans.py
08	🌳 Hierarchical Clustering	MainClusteringDendrogram.m	MainClusteringDendrogram.py
09	🖼️ Image Compression	MainImageCompress.m	MainImageCompress.py
10	🦋 Lorenz Neural Network	MainRegressionNN.m	MainRegressionNN.py
11	🌪️ Rössler Echo State Network	MainRosslerForecast.m	MainRosslerForecast.py

Each example illustrates a particular mathematical or computational concept through a numerical experiment and graphical visualization.

⸻

🚀 Getting Started

📥 1. Clone the Repository

git clone https://github.com/americocunhajr/ShortCourse-AI-ML-Essentials.git
cd ShortCourse-AI-ML-Essentials

🔶 2. Running the MATLAB Examples

Open MATLAB and set the working directory to the repository root.

Execute an example by entering its name in the Command Window:

MainRegressionFreeFall

Some examples may require additional MATLAB toolboxes.

🐍 3. Running the Python Examples

Requirements: Python 3.10 or newer is recommended.

The implementations use the following scientific computing libraries:

* 🔢 NumPy: array operations and numerical linear algebra.
* 🧪 SciPy: optimization, numerical integration, and scientific algorithms.
* 📊 Matplotlib: scientific visualization.
* 🤖 scikit-learn: machine learning algorithms.
* 🖼️ Pillow: image processing.

Navigate to the Python directory:

cd python

Create a virtual environment:

python -m venv .venv

Activate it on Linux/macOS:

source .venv/bin/activate

Or on Windows PowerShell:

.venv\Scripts\Activate.ps1

Install the required packages:

python -m pip install -r requirements.txt

Execute your first example:

python MainRegressionFreeFall.py

For a faster demonstration of the Echo State Network:

python MainRosslerForecast.py --quick

The --quick option uses reduced computational settings suitable for classroom demonstrations.

⸻

🧭 Suggested Learning Path

A possible progression through the course is:

Stage	Subject	Focus
1️⃣	Regression	Mathematical modeling and optimization
2️⃣	Classification	Supervised learning and decision boundaries
3️⃣	Clustering	Unsupervised learning and data organization
4️⃣	Dimensionality Reduction	Matrix methods and information compression
5️⃣	Neural Networks	Nonlinear approximation and surrogate modeling
6️⃣	Reservoir Computing	Learning and forecasting nonlinear dynamics

The sequence moves from elementary numerical modeling toward advanced data-driven methods for dynamical systems.

⸻

🔬 Numerical Reproducibility

The Python implementations reproduce the intended mathematical experiments of the MATLAB examples, but the numerical results are not necessarily identical.

Differences may arise from:

* Different numerical optimization and linear algebra implementations.
* Solver configurations and convergence tolerances.
* Random initialization and sampling.
* Floating-point arithmetic.
* Differences between MATLAB and Python machine learning libraries.

In particular, the Python neural-network example for the Lorenz system uses MLPRegressor from scikit-learn. Its architecture and training procedure do not exactly reproduce the corresponding MATLAB implementation.

For chaotic dynamical systems, sensitive dependence on initial conditions makes long-term pointwise trajectory agreement difficult, even when the learned model captures relevant dynamical features.

Short-horizon predictive accuracy and long-term statistical behavior should therefore be evaluated separately.

✅ Running Automated Tests

The Python implementation includes basic automated tests.

Install pytest:

python -m pip install pytest

Run:

python -m pytest -q

These tests verify selected implementation and execution properties; they do not establish complete numerical equivalence between MATLAB and Python.

⸻

👩‍🎓 Intended Audience

The course is intended for:

* 🎓 Undergraduate and graduate students.
* 🔬 Researchers in applied mathematics and computational science.
* ⚙️ Engineers interested in data-driven modeling.
* 💻 Scientific programmers exploring machine learning.
* 📊 Practitioners interested in the mathematical foundations of AI.

Previous knowledge of calculus, linear algebra, and basic programming is recommended.

⸻

👨‍🏫 Instructor

Americo Cunha Jr.

Research Faculty — Laboratório Nacional de Computação Científica (LNCC)

Professor of Applied and Computational Mathematics — Universidade do Estado do Rio de Janeiro (UERJ)

🌐 Website: americocunha.org

🔗 GitHub: americocunhajr

📚 Lattes: Academic CV

Research interests include nonlinear dynamics, uncertainty quantification, computational mechanics, inverse problems, reduced-order modeling, scientific machine learning, and data-driven modeling.

⸻

🤝 Contributions and Feedback

Comments, suggestions, and corrections are welcome.

If you encounter a bug, identify a numerical inconsistency, or have a suggestion for improving an example, please open a GitHub Issue.

When reporting a problem, include the example name, software environment, and sufficient information to reproduce the behavior.

⸻

📚 Citation

If you use these materials in teaching, research, or other academic activities, please acknowledge the repository:

Cunha Jr., A. Artificial Intelligence and Machine Learning Essentials. Short-course computational examples. GitHub repository. https://github.com/americocunhajr/ShortCourse-AI-ML-Essentials

BibTeX

@misc{CunhaAIMLEssentials,
  author       = {Cunha Jr., Americo},
  title        = {Artificial Intelligence and Machine Learning Essentials},
  howpublished = {\url{https://github.com/americocunhajr/ShortCourse-AI-ML-Essentials}},
  note         = {Short-course computational examples}
}

⸻

⚖️ License

No software license is specified in this README.

If public redistribution, modification, and reuse are intended, an explicit LICENSE file should be included in the repository.

⸻

🤖 Artificial Intelligence & Machine Learning Essentials

Mathematical foundations, computational experiments, and data-driven modeling.

Americo Cunha Jr. | LNCC & UERJ | Brazil