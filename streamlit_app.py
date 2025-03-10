import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_iris
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="EDA IRIS DATASET",
    layout="wide",
    initial_sidebar_state="expanded")

try:
    st.cache_data.clear()
    st.cache_resource.clear()
except:
    pass

@st.cache_data(ttl=0.5*3600)
# cache the dataframe, so it’s only loaded once when the app starts.
def load_data():
    iris = load_iris()
    dataframe = pd.DataFrame(iris.data, columns=iris.feature_names)    
    # Add the species column
    dataframe['species'] = iris.target_names[iris.target]

    return dataframe

# Apply background gradient based on species
def color_species(val):
    color = 'lightgreen' if val == 'setosa' else 'lightblue' if val == 'versicolor' else 'pink'
    return f'background-color: {color}'

def show_dataset(dataframe, rows):    
    result = dataframe.groupby('species').head(rows)
    # Style the DataFrame
    styled_result = result.style.map(color_species, subset=['species']).set_properties(**{'text-align': 'center'})
    
    return styled_result

def plot_heatmap(dataframe):    
    # Calculate correlation for numeric columns only
    corr = dataframe.select_dtypes(include=['float64', 'int64']).corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the matplotlib figure
    fig_iris = plt.figure(figsize=(6, 4))
    
    # Create the heatmap with improved aesthetics
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                annot=True,
                fmt=".2f",  # Format annotations to 2 decimal places
                cmap='viridis',  # Use a more visually appealing colormap
                linewidths=0.5,  # Add lines between cells
                mask = mask,
                annot_kws={"size": 8}
                )  # Adjust annotation font size
   
    # Rotate x-axis labels and reduce font size
    plt.xticks(rotation=45, ha='right', fontsize=8)  # Reduce x-axis font size to 10
    plt.yticks(rotation=0, ha='right', fontsize=8)  # Reduce y-axis font size to 10
    
    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent clipping of labels    
    return fig_iris

def plot_iris(dataframe):
    # Create the pairplot
    pairplot = sns.pairplot(dataframe, hue='species', height=2.5, aspect=1.2)
    
    # Access the legend and modify its properties
    sns.move_legend(
        pairplot, "lower center",
        bbox_to_anchor=(0.5, 1), ncol=3, title=None, frameon=False,
    )
    return pairplot

def my_linspace (min_value, max_value, steps):
    diff = max_value - min_value
    return np.linspace (min_value - 0.1 * diff, max_value + 0.1 * diff, steps)

def my_mask (value, selected):
    min_value = 1.7e+308
    for i in range(len(value)):
        if selected[i] and value[i] < min_value:
            min_value = value[i]
    for i in range(len(value)):
        if not selected[i]:
            value[i] = min_value
    return value


@st.cache_data(ttl=0.5*3600)
def load_iris2():
    return load_iris()

def plot_svm():
    iris = load_iris2()
    x = iris.data[:,:2]
    y = iris.target

    steps = 200
    x0 = my_linspace(min(x[:,0]), max(x[:,0]), steps)
    x1 = my_linspace(min(x[:,1]), max(x[:,1]), steps)
    xx0, xx1 = np.meshgrid(x0, x1)
    mesh_data = np.c_[xx0.ravel(), xx1.ravel()]

    color = ['red', 'green', 'blue']
    y_color = [color[i] for i in y]
    
    # Create an SVM classifier
    svm_classifier = SVC(kernel=kernel_value, C=c_value, probability=True, random_state=42, decision_function_shape='ovo')
    
    # Train the model
    svm_classifier.fit(x, y)

    pred = svm_classifier.predict(mesh_data)
    deci = svm_classifier.decision_function(mesh_data)
    prob = svm_classifier.predict_proba(mesh_data)

    n_class = 3
    mesh_deci = np.zeros((steps * steps, n_class))
    mesh_prob = np.zeros((steps * steps, n_class))

    # Plot decision boundaries for each pair of classes
    j = 0
    for k in range(n_class):
        for l in range(k + 1, n_class):
            mesh_deci[:, k] += deci[:, j]  # Update decision function for class k
            mesh_deci[:, l] -= deci[:, j]  # Update decision function for class l 
            j += 1

    # Apply mask to probabilities and decision values
    for j in range(n_class):
        prob[:, j] = my_mask(prob[:, j], pred == j)
        mesh_deci[:, j] = my_mask(mesh_deci[:, j], pred == j)
    
    # Reshape mesh_deci and prob for plotting
    mesh_deci = mesh_deci.reshape(steps, steps, n_class)
    mesh_prob = prob.reshape(steps, steps, n_class)

    svmfig = plt.figure(figsize = (12, 6))
    contour_color = [plt.cm.Reds, plt.cm.Greens, plt.cm.Blues]

    col1, col2 = st.columns([1,1], gap="small")
    with col1:    
        # Plot decision function contours  
        plt.subplot(1, 2, 1)     
        for j in range(n_class):
            plt.contourf(xx0, xx1, mesh_deci[:, :, j], 20, cmap=contour_color[j], alpha=0.3)
        plt.scatter(x[:, 0], x[:, 1], c=y_color, edgecolors='k')
        plt.title(f'Decision Function (C: {c_value} | Kernel: {kernel_value})')
    with col2:    
        # Plot probability contours
        plt.subplot(1, 2, 2)
        for j in range(n_class):
            plt.contourf(xx0, xx1, mesh_prob[:, :, j], 20, cmap=contour_color[j], alpha=0.3)
        plt.scatter(x[:, 0], x[:, 1], c=y_color, edgecolors='k')
        plt.title(f'Class Probabilities (C: {c_value} | Kernel: {kernel_value})')

    return svmfig

def evaluation(y_test, y_pred):
    df = load_data()
    class_names = df['species'].unique()

    col1, col2 = st.columns([1,1], gap="small")
    with col1:
        # Compute the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Display the confusion matrix with label names
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        st.pyplot(fig)
   
    with col2:            
        # # Display accuracy score
        # st.subheader("Accuracy Score")
        # accuracy = accuracy_score(y_test, y_pred)
        # st.write(f"The accuracy of the model is: **{accuracy:.2f}**")

        # Calculate metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average='micro')
        recall = metrics.recall_score(y_test, y_pred, average='micro')
        f1 = metrics.f1_score(y_test, y_pred, average='micro')
        
        st.write("\n" + "=" * 40)
        st.write("Performance Metrics".center(40))
        st.write("=" * 40)
        st.write(f"{'Accuracy:':^20}{accuracy:.4f}")
        st.write(f"{'Precision:':^20}{precision:.4f}")
        st.write(f"{'Recall:':^20}{recall:.4f}")
        st.write(f"{'F1-Score:':^20}{f1:.4f}")
        st.write("=" * 40)

    return

def predict_GaussianNB(dataframe):  
    
    # Split the data into features (X) and target (y)
    X = dataframe.drop(columns=['species'])  # Features
    y = dataframe['species']  # Target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Naive Bayes classifier
    nb_classifier = GaussianNB()
    
    # Train the classifier
    nb_classifier.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = nb_classifier.predict(X_test)
    
    return evaluation(y_test, y_pred)

def predict_SVM(dataframe):

    # Split the data into features (X) and target (y)
    X = dataframe.drop(columns=['species'])  # Features
    y = dataframe['species']  # Target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create an SVM classifier
    svm_classifier = SVC(kernel=kernel_value, C=c_value, random_state=4, decision_function_shape='ovo')
    
    # Train the model
    svm_classifier.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = svm_classifier.predict(X_test)

    return evaluation(y_test, y_pred)

# Sidebar for User Inputs
with st.sidebar:
    st.title("DataFrame Options")
    st.write("`Created by: Andre`")
    num_row = st.number_input("Dataset Rows to View", value=3)
    st.markdown("---")
    c_value = st.selectbox("SVM c_value:", [1.0, 5.0, 10.0, 50.0])
    kernel_value = st.selectbox("SVM kernel_value:", ["rbf", "linear", "poly", "sigmoid"])

# Main Page for Output Display
st.title("IRIS Dataset Snapshot")
st.info("Toggle at the sidebar to view number of rows")
dataframe = load_data()

# Convert the DataFrame to HTML and center the text
html = show_dataset(dataframe, num_row).to_html(index=False)
centered_html = f"""
<style>
table {{
    margin-left: auto;
    margin-right: auto;
    text-align: center;
}}
th, td {{
    text-align: center !important;
}}
</style>
{html}
"""
# Display the centered table using st.markdown()
st.markdown(centered_html, unsafe_allow_html=True)

st.title("Heatmap and Correlation Plot")
col1, col2 = st.columns([1,1], gap="small")
with col1:
    st.subheader("Heatmap of Iris Dataset Features")
    heatmap_fig_iris = plot_heatmap(dataframe)
    st.pyplot(heatmap_fig_iris)

with col2:
    st.subheader("Correlation Plot")
    pairplot_fig_iris = plot_iris(dataframe)
    st.pyplot(pairplot_fig_iris)

st.title(f"Prediction with SVM with kernel={kernel_value} and C={c_value}")
predict_SVM(dataframe)

fig_svm = plot_svm()
st.pyplot(fig_svm)

st.title("Prediction with GaussianNB")
predict_GaussianNB(dataframe)

try:
    st.cache_data.clear()
    st.cache_resource.clear()
except:
    pass
