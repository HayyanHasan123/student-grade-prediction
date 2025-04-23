import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("C:\\Users\\T L S\\Documents\\student\\student-mat.csv", sep=";")

print(df.info()) # Tells us about no of colums and rows
print(df.describe()) # Tells us about the data
print(df.isnull().sum()) # Tells us how many unfilled areas there are


# One-Hot Encoding to convert string based columns into machine readable columns for Correlation Analysis
one_hot_enc = pd.get_dummies(df, drop_first = True)
co_r_matrix = one_hot_enc.corr()
# +1 & -1 = Strong correlation
# Near 0 = Weak Correlation
print(co_r_matrix)


# Using Matplotlib for graphical visualization of correlation matrix
plt.figure(figsize = (12,8))
sns.heatmap(co_r_matrix,
            annot = False,
            cmap = "coolwarm",
            fmt =".2f",
            cbar = True
)
plt.title("Grade Prediction Correlation Matrix Heatmap")
plt.show()





# Now plotting scattered graph for comparing G2 and G3 using a Regression Line
# Setting up a new figure for the scattered graph

plt.figure(figsize = (12,8))

# Scattered Regression Line graph of G2 against G3
# G2 as X axis
# G3 as Y axis
sns.regplot(
    x = "G2",
    y = "G3",
    data = df,
    scatter_kws = {
        's' : 20,
        'color' : 'blue',
        'alpha' : 0.6
    },
    line_kws = {
        'color' : 'red',
        'linewidth' : 1 
    }
)

# Now labeling these axis according to their respective data for clarity
plt.xlabel("Second Period Grades(G2)")
plt.ylabel("Third Period Grades(G3)")

# Labeling the graph itself
plt.title("Scatter Plot Graph of G2 Vs G3 Using Regression Line")

# For Proper Layout
plt.tight_layout()
plt.show()




# G1 VS G3
plt.figure(figsize = (12,9))
sns.regplot(
    x = "G1",
    y = "G3",
    data = df,
    scatter_kws = {
        's' : 20,
        'color' : 'blue',
        'alpha' : 0.6
    },
    line_kws = {
        'color' : 'red',
        'linewidth' : 1 
    }
)
plt.xlabel("First Period Grade(G1)")
plt.ylabel("Third Period Grade(G3)")
plt.title("Scatter Plot Graph of G1 VS G3 Using Regression Line")
plt.tight_layout()
plt.show()




# Failure Vs G3
plt.figure(figsize = (12,8))
sns.regplot(
    x = "failures",
    y = "G3",
    data = df,
    scatter_kws = {
        's' : 20,
        'color' : 'blue',
        'alpha' : 0.6
    },
    line_kws = {
        'color' : 'red',
        'linewidth' : 1 
    }
)
plt.xlabel("Number of Past Class Failures(Failures)")
plt.ylabel("Third Period Grade(G3)")
plt.title("Scatter Plot of Failure Vs G3 Using Regression Line")
plt.tight_layout()
plt.show()




# Absences vs G3
plt.figure(figsize = (12,8))
sns.regplot(
    x = "absences",
    y = "G3",
    data = df[df["absences"] <= 30], 
    scatter_kws = {
        's' : 20,
        'color' : 'blue',
        'alpha' : 0.6
    },
    line_kws = {
        'color' : 'red',
        'linewidth' : 1 
    }
)
plt.xlabel("Number os Absences(Absences)")
plt.ylabel("Third Period Grade(G3)")
plt.title("Scatter Plot of Abscences Vs G3 Using Regression Line")
plt.tight_layout()
plt.show()




# Model Training and Evaluation
# In this section, we will train the Linear Regression model on the training dataset
# and evaluate its performance on both training and testing datasets.
# The data will be split into 2 parts, 80% for Training & 20% for Testing

x = df[['G1' , 'G2' , 'failures' , 'absences']]
y = df['G3']

# Now Splitting the Data
#Using any Number as Random State so the model's results remain consistent every time the code is run
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2) 

# Creating and Training the Model
model = LinearRegression()
model.fit(x_train, y_train)

# Model Coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)




# Now for predicting G3
y_predict = model.predict(x_test)
# Checking the authenticity of the model
mean_s_e = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print("Mean Squared Error: ", mean_s_e)
print("R-Squared Score: ", r2 )




# For visual comparision between real and predicted values
plt.figure(figsize = (12,8))
plt.scatter(
    y_test,
    y_predict,
    color = 'blue',
    alpha = 0.6
)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color = "red",
         linewidth = 2
)
plt.xlabel("Actual Third Period Grades")
plt.ylabel("Predicted Third Period Grades")
plt.title("Actual Grades Vs Predicted Grades")
plt.show()
