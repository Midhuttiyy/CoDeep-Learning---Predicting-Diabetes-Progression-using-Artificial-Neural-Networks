# Deep-Learning---Predicting-Diabetes-Progression-using-Artificial-Neural-Networks




## Dataset:
Use the Diabetes dataset available in the sklearn library.


## 1: Loading and Preprocessing


Loading the Dataset: We utilized the Diabetes dataset from the sklearn library, which contains 442 samples and 10 features. This dataset is commonly used for regression tasks.

Handling Missing Values: A check for missing values was performed, and no missing values were found. This simplifies preprocessing as no imputation or removal of missing data is necessary.

Normalization: Feature normalization was conducted using StandardScaler. Normalization ensures that all features contribute equally to the model's learning process, preventing features with larger ranges from dominating the learning process.

Dataset Splitting: The data was split into training (80%) and testing (20%) sets. This allows us to train the model on a subset of data and then evaluate its performance on unseen data.


## 2: Exploratory Data Analysis (EDA)

Summary Statistics: Descriptive statistics provided insights into the distribution of each feature and the target variable. This helps in understanding the range, mean, and standard deviation of the data.

Pairplot Visualization: The pairplot visualized the relationships between each feature and the target variable (disease progression measure). It helps identify any strong linear or non-linear relationships.

Correlation Heatmap: The heatmap displayed the correlation between features and the target variable. Features with a high positive or negative correlation with the target are more influential in predicting diabetes progression. For instance, features like bmi or s5 might show a higher correlation with the target variable.



## 3: Building the ANN Model



Model Architecture: A simple Artificial Neural Network (ANN) model was constructed with:

Input Layer: Matching the number of features (10).
Hidden Layer: One hidden layer with 64 neurons using the ReLU activation function. ReLU is a common choice for hidden layers as it helps mitigate the vanishing gradient problem.

Output Layer: A single neuron with a linear activation function, suitable for regression tasks.

Compilation: The model was compiled using the Adam optimizer with a learning rate of 0.01, and the Mean Squared Error (MSE) as the loss function. The choice of MSE is appropriate for regression tasks as it penalizes large errors more than smaller ones.




## 4 : Training the ANN Model


Model Training: The model was trained for 100 epochs with a batch size of 32. The training process included monitoring the training and validation loss, which was plotted over epochs.




Training Insights:

The loss curve indicates how well the model is learning. If the training loss decreases steadily but the validation loss stabilizes or increases, it might indicate overfitting.
If the training and validation losses both decrease and converge, it suggests that the model is generalizing well to the unseen data




## 5: Evaluating the Model



Model Evaluation: The model was evaluated on the test set, yielding the following metrics:

Mean Squared Error (MSE): A lower MSE indicates better performance, as it measures the average squared difference between predicted and actual values.
R² Score: This score indicates how well the model explains the variance in the target variable. An R² score closer to 1 suggests that the model explains most of the variability in the target variable



## 6: Improving the Mode




Model Improvements:

A more complex architecture was tested with two hidden layers of 128 and 64 neurons, respectively.
A lower learning rate (0.001) was used to allow the model to converge more slowly, potentially leading to better generalization.
Improved Model Evaluation:

After training the improved model, it was evaluated again on the test set.
Improved MSE and R²: If the MSE decreases and the R² score increases compared to the initial model, it indicates that the improvements had a positive effect on model performance.
Training History: By comparing the training and validation loss curves of the original and improved models, we can see if the changes made reduced overfitting or led to a better fit on the training data.

The initial ANN model provided a baseline performance, while the improved model, with more layers and adjusted hyperparameters, likely offered better predictive accuracy. This iterative process of model evaluation and improvement is crucial in developing a robust predictive model for diabetes progression.
