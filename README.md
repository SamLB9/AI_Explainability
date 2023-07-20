# AI_Explainability

Artificial Intelligence (AI) makes it possible to harness huge volumes of data to perform complex tasks, usually very efficiently. However, AI models, and more specifically Machine Learning and even Deep Learning models, are often compared to "black boxes" due to their opaque operation, not necessarily understandable by humans. The complexity of these models is therefore a strength, as it enables us to respond to problems better than with simpler models, but it is also a weakness, as it makes them difficult to interpret. However, in certain critical fields such as medical diagnostics or autonomous driving, where human lives may be at stake, control and understanding of the decision-making mechanisms of these models is essential.

Before going into detail, we'll define the terms explicability and interpretability of models.

### Interpretability:
The interpretability of an ML model is linked to its simplicity and overall comprehensibility. A model is interpretable when it is easy for humans to understand how it works, even if its explanations are not necessarily exhaustive. An interpretable model must be simple and have a clear structure, so that users can analyze and trust it without needing in-depth knowledge of machine learning. Interpretability is often favored when simplicity and overall understanding are more important than fine-grained explanation of predictions.

### Explicability:
The explicability of an ML model refers to its ability to provide a clear and understandable explanation of how it makes decisions. This means you can understand why the model has predicted a certain thing or taken a certain action. Explainability is essential when you want to know which features or variables are important to the model and how they influence predictions. An explanatory model enables you to understand the decision-making process, which is crucial in sensitive areas such as healthcare or finance.

Research in this field is very active, and companies are investing more and more in these issues.

AI explicability or interpretability aims to make the operation and results of models more intelligible and transparent to humans, without compromising on performance.
However, the difficulty today is that the more precise the models, the less explicable they become. So we're looking for a compromise. Even if for many data scientists, precision seems to be the supreme indicator of model quality, it's important to understand that it isn't everything. In deep learning, for example, explaining models is very difficult. Finding an algorithm as efficient as deep learning that is also 100% explainable is (for the moment?) utopian.

This need for explicability concerns different players, each with their own motivations. Firstly, for the Data Scientist who develops the model, a better understanding can help correct certain problems and improve the model. On the business side, explainability can be a means of assessing how well the model matches the company's strategy and objectives. Explanability can also be used to test the model's robustness, reliability and potential impact on customers. Finally, it can enable the customer who would be the object of a decision by a system based on an AI model to be informed of the impact of this decision and the potential actions possible to modify it.

![image](https://github.com/SamLB9/AI_Explainability/assets/106078401/08a09646-078c-4e6b-b175-3af1c75b029f)

In certain fields, such as health or law, the use of machine learning models that cannot be explained raises ethical questions and can prove dangerous. This aspect should be much more closely regulated. It could be interesting to create a text similar to the RGPD, but for explicability, which is an issue at least as sensitive as user data protection.

First, we'll look at methods for understanding models, and then we'll see which models are the most interpretable and explainable.

## How to explain a machine learning model and how it works?

### 1) Permutation Feature Importances

Permutation Feature Importances is a widely used analysis method for assessing the importance of features in a Machine Learning model. This technique enables us to understand which features are the most influential in the model's predictions.

Here's how the Permutation Feature Importances method works:
1) Initial model training: First, you train the model on your training data with all the features included.
2) Performance evaluation: Next, you evaluate the model's performance using an appropriate evaluation metric (such as accuracy for classification or mean square error for regression) on a test dataset.
3) Feature permutation: Now, for each feature, you will randomly permute the values of that feature in the test dataset. This means that you will mix the values of the characteristic without changing the other characteristics.
4) Evaluation after swapping: After swapping a characteristic's values, you evaluate the model's performance again on the test dataset.
5) Importance calculation: The importance of a characteristic is calculated by comparing model performance before and after swapping. If performance decreases significantly after permutation, this means that the feature is important for model predictions. On the other hand, if performance remains more or less the same after permutation, this means that the feature has little influence on predictions.
6) Repeat: Repeat steps 3 to 5 for all features to obtain the importance of each feature.

Here's a Permutation Feature Importances graph example:
![image](https://github.com/SamLB9/AI_Explainability/assets/106078401/24fc3a08-33f1-41b5-9592-ebe3ef145327)

Permutation Feature Importances offer a simple yet powerful method for assessing the importance of features in a model. This enables us to understand which features contribute most to the model's predictions, and which are the most informative for the given task. This information can be invaluable in making informed decisions when analyzing model results, and in improving model performance where necessary.

### 2) Shapley Values:
Shapley Values are a concept from game theory, which has been adapted to explain the predictions of Machine Learning models in a fair and consistent way. This notion is used to assign a value to each characteristic (or attribute) of an observation, measuring its marginal contribution to the prediction.

Imagine you have a Machine Learning model that predicts the price of a house as a function of various characteristics such as surface area, number of bedrooms, location and so on. Shapley Values allow you to understand how each feature contributes to the price prediction.

Here's how Shapley Values work:
1) Fix all other characteristics: For a given observation (a specific house in our example), you keep the other characteristics fixed at their actual values.
2) Vary the order of features: You can vary the order in which features are taken into account to make the prediction. For example, you might start by considering the surface area, then add the number of bedrooms, then the location, and so on.
3) Calculate marginal contributions: For each feature, you calculate the marginal contribution it makes to the model prediction when added to the set of features already taken into account. This measures the importance of each feature in relation to its interaction with the others.
4) Calculate Shapley Values: Once you have all the marginal contributions for each feature, you combine these values using a specific formula from game theory, called the Shapley formula. This formula assigns a Shapley Value to each feature, representing the average contribution of that feature over all possible combinations of feature orderings.

Shapley Values help to explain predictions in a consistent and fair way, by assigning values to each feature that measure its relative importance in the prediction process. These values are also additively consistent, meaning that the sum of the Shapley Values for all features is equal to the difference between the model's prediction for the given observation and the model's average prediction over the entire data set. This property makes Shapley Values particularly useful for the explicability of Machine Learning models.

### 3) SHAP Summary Plot:

The SHAP Summary Plot (or SHAP Summary Chart) is a visualization used to interpret and understand the predictions of a Machine Learning model based on the SHAP (SHapley Additive exPlanations) method. This technique explains the importance of each feature (or variable) in the model and its impact on predictions.

Here's how the SHAP Summary Plot works:
1) Calculation of SHAP values: For each data instance in the test set (or in a selection of samples), SHAP values are calculated for each feature. SHAP values represent the contribution of each feature to each specific prediction. In other words, they measure the effect of each characteristic on the difference between the model prediction and a reference value (for example, the mean model prediction).
2) Aggregation of SHAP values: Next, SHAP values are aggregated across all test samples. This gives an overall view of the importance of each feature for all model predictions.
3) SHAP Summary Plot construction: The SHAP Summary Plot is a graphical representation of these aggregated values. Features are arranged in descending order of importance on the horizontal (or vertical) axis, and the corresponding SHAP values are displayed next to each feature. Positive SHAP values indicate that the presence of that feature has a positive influence on the prediction, while negative SHAP values indicate a negative influence.
4) Average effect : A vertical (or horizontal) line can be added to the graph to represent the overall average effect of all features on model predictions. This provides a better understanding of how all features contribute to the overall prediction.

Here's a SHAP Summary Plot example:
![image](https://github.com/SamLB9/AI_Explainability/assets/106078401/4cc608f9-9b71-44ae-82d8-0dc611471a3f)

The SHAP Summary Plot provides a synthetic and comprehensible view of the importance of features in the model. It helps to understand which features have the greatest impact on predictions, and in which direction this impact manifests itself. It's a powerful tool for interpreting ML models, enabling users to better understand how the model makes its decisions, and which features are most influential in this process.

### 4) LIME (Local Interpretable Model-agnostic Explanations):

LIME (Local Interpretable Model-agnostic Explanations) is a technique used to explain the predictions of Machine Learning models, particularly those that are considered "black boxes" and difficult to interpret.
The principle behind LIME is to locally create a simplified, interpretable model around a specific data instance to explain how the original model makes its decision for that instance. In other words, LIME seeks to explain the predictions of the complex model using a simpler, locally explicit model.

Here's how LIME works:
1) Choose an instance: You choose a data instance for which you want to explain the model prediction. This instance can be a specific input data point (for example, an image, text or numerical vector) for which you want to understand why the model gave a certain prediction.
2) Generation of neighboring data: Around this chosen instance, LIME generates new similar data by performing small perturbations and variations on the characteristics of the original instance.
3) Calculation of local predictions: The complex Machine Learning model is used to predict the outputs for each new data item generated in the previous step. This results in a series of local predictions for each variation in the characteristics of the chosen instance.
4) Building a local model: Using this new data and its predictions, LIME builds a simpler model, such as a linear regression, to explain locally how features affect the predictions of the original model around the selected instance.
5) Local model interpretation: The local model, being simpler and more explicit, can be used to explain how each feature influences the prediction for that specific instance. This provides a better understanding of the reasoning behind the original model's prediction for that particular instance.

Example of a local explanation generated by the LIME method:

![image](https://github.com/SamLB9/AI_Explainability/assets/106078401/d2343723-6ca3-494f-869b-6de413677bfa)

The advantage of LIME is that it can be used with any Machine Learning model, as it does not depend on the internal structure of the complex model. This makes it "model-agnostic", i.e. it can provide explanations for different models without needing to know their specific architecture. LIME is particularly useful for explaining high-dimensional models such as deep neural networks, which may otherwise be difficult to interpret.

### 5) Anchors:

The Anchors method has been developed to overcome some of LIME's problems, notably concerning the generalizability of explanations, which is not clearly defined. Anchors is an interpretability technique for Machine Learning models, developed to enable a finer-grained understanding of the decisions made by these models. The concept of Anchors was presented in the research paper "Anchors: High-Precision Model-Agnostic Explanations" by Ribeiro et al. in 2018.

The aim of Anchors is to provide concise, easy-to-understand explanations for a model's predictions, by identifying simple rules about the input features that lead to a specific prediction. In other words, Anchors seeks to answer the question: "Why did the model make this particular decision?".

Here's how Anchors work:
1) Explanatory rules: Anchors consist of simple logical rules about input features. For example, for a fraud prediction model, an explanatory rule might be "If the transaction amount is greater than $1,000 and the transaction location is abroad, then the model predicts fraud".
2) Binary features: Anchor rules are expressed as binary features, meaning that each feature is considered either present or absent. This simplifies explanations while retaining a certain expressiveness.
3) Anchor Search: To find an anchor for a given prediction, the algorithm explores different combinations of binary features to find the simplest and most precise rule that explains the prediction. The aim is to find a rule that is restrictive enough to be an accurate explanation, yet as simple as possible.
4) High accuracy: Anchors are designed to have high accuracy, which means that they explain model predictions with a high degree of accuracy. They are designed to be reliable and informative.

Example of a local explanation generated by the ancres method:

![image](https://github.com/SamLB9/AI_Explainability/assets/106078401/62ac21db-a884-48ee-a69e-b55c9fe9a5c8)

The explanation given for the chosen customer indicates the variables that anchor the churn prediction: a rather low seniority (less than 29 months), a type of electronic payment, a monthly contract, etc. The added value of this method is linked to an additional piece of information, the coverage. In this case, 8% of customers have the characteristics defined by the anchor.

Anchors are considered "model-agnostic", meaning they can be applied to different types of Machine Learning models without requiring specific knowledge of their internal architecture. This interpretive approach can be very useful in understanding how models make decisions, and in gaining the trust of users and stakeholders by providing clear and understandable explanations.

### 6) Counterfactual explanations:

In contrast to the previous method, which focuses on the variables that anchor predictions, the counterfactual explanation method looks for those whose change modifies the prediction. Counterfactual explanations" are an approach to the interpretability of Machine Learning models that aims to explain the model's predictions by proposing hypothetical scenarios in which a prediction would have been different.

The fundamental idea behind counterfactual explanations is to provide an answer to the question, "What would have had to be different in the input data for the model to produce a different prediction?"

Here's how it works:
1) Existing data point: You have a specific data point for which you want to explain the model's prediction. For example, this could be an image, a text description, or feature values for a prediction model.
2) Current prediction: The model has already made a prediction for this data point, and you know the model's current output for this input.
3) Counterfactual scenario: The counterfactual explanation will now seek to find a hypothetical scenario where a feature or set of features is changed in such a way as to produce a different prediction. This means that you will change certain values in the input data, while keeping the rest of the data constant, until the model predicts a different output.
4) Presentation of the explanation: Once the counterfactual scenario has been found, the explanation can be presented by showing how specific characteristics have been modified, leading to a different prediction from the model.

Example of a counterfactual explanation:

![image](https://github.com/SamLB9/AI_Explainability/assets/106078401/9f34fe28-4f5a-49e3-b150-e9b75c86ae1a)

Still on the subject of the chosen customer, a first counterfactual explanation consists in the modification of his type of contract and his type of payment; with these two changes, the model predicts non-cancellation. The same applies if only the customer's seniority is modified, from 13 to 65 months.

Counterfactual explanations are useful for understanding how individual characteristics influence model predictions, and what conditions would be necessary to obtain an alternative prediction. They highlight the factors that have the greatest impact on the model's results, and help identify the key points that differentiate the model's predictions for different data instances.

Counterfactual explanations are particularly useful in critical areas where decisions based on model predictions have important consequences, such as healthcare or financial lending systems, as they enable users to better understand the underlying reasons behind model predictions.

### 7) Partial Dependence Plot (PDP) 
Partial Dependence Plot (PDP) is a method used to understand how a specific variable influences the predictions of a Machine Learning model. It allows us to observe the overall effect of this variable on predictions, taking into account interactions with other variables.

Let's imagine we have a prediction model for house prices, and we want to know how the size of the house affects predictions.

Here's how the Partial Dependence Plot works:
1) Fix all other variables: As in the other techniques, keep the other characteristics fixed for each data instance.
2) Vary the area: You'll now vary the area of the house for this data instance, taking different possible area values.
3) Average predictions: At each area value, make predictions with the model, but this time, rather than looking at individual predictions as in ICE, you record the average of predictions for all data instances when the area is set to this value.
4) Plot the PDP graph: Finally, plot the PDP graph, where the x-axis represents the different area values, and the y-axis represents the average predictions for each area value. The graph shows how area influences model predictions on average.

Here's an PDP graph example:
![image](https://github.com/SamLB9/AI_Explainability/assets/106078401/0394c542-1f3d-4e3d-ac9e-a22f463614b5)

The Partial Dependence Plot provides a global, average view of the effect of a variable on model predictions, taking into account interactions with other variables. This provides a better understanding of the relationship between the variable under study and the model's predictions on the whole data set. PDP is useful for identifying general model trends and relationships between features without focusing on individual predictions.

### 8) Individual Conditional Expectation(ICE):
Individual Conditional Expectation (ICE) is a technique used to understand how a particular variable affects the predictions of a Machine Learning model. Essentially, it involves decomposing the effect of that variable on predictions, by showing how the values of that variable individually influence the results of the model.

To better understand, imagine you have a prediction model, for example one that predicts the price of a house based on its characteristics such as surface area, number of bedrooms, etc. You want to understand how the surface area of the house affects the price of the house. You want to understand how the size of the house affects price predictions.

With Individual Conditional Expectation, you'll take the following steps:
1) Fix all other variables: For each data instance (i.e. a specific house in our example), keep the other characteristics fixed, such as number of bedrooms, location, etc.
2) Vary the area: You will now vary only the area of the house for this data instance, taking different possible area values.
3) Observe predictions: At each area value, make predictions with the model and record the results (i.e. the predicted prices). You will obtain a series of predictions corresponding to different area values.
4) Plot the ICE graph: Finally, plot the ICE graph, which shows on the x-axis the different area values you've tested, and on the y-axis, the corresponding predictions. The graph will show you how the predictions change as the area of the house varies.

Here's an ICE graph example:
![image](https://github.com/SamLB9/AI_Explainability/assets/106078401/8aec28fe-8b2b-4bd5-8942-5f09b0a50b40)

ICE allows you to visualize how each individual variable affects the model's predictions, regardless of interactions with other variables. This can provide valuable information on the relationship between features and model predictions, and enable a better understanding of the factors influencing prediction results.

### 9) Accumulated Local Effects (ALE):
Accumulated Local Effects (ALE) is a technique used to understand how variables in a Machine Learning model influence predictions. It provides a global, accumulated view of the effect of a specific variable on the model's predictions, as seen through ALE.

Imagine you once again have a prediction model that estimates the price of a house based on different characteristics. This time, you want to understand how the size of the house affects the predictions, but in a more global way.

Here's how Accumulated Local Effects works:
1) Set all other variables: As with ICE, you keep the other characteristics fixed for each data instance.
2) Vary the surface area: You will now vary the surface area of the house for this data instance, taking different possible surface area values, as in ICE.
3) Calculate the local effect: At each area value, you calculate the local effect of the area on the model prediction. The local effect is simply the difference between the model prediction for a certain area value and the average model prediction for all area values.
4) Accumulate local effects: You then accumulate these local effects to obtain a global view of the effect of area on model predictions. For each area value, you add the local effect to the accumulation of previous local effects.
5) Draw the ALE graph: Finally, draw the ALE graph, where the x-axis represents the different area values, and the y-axis represents the accumulated effects. The graph will show you how the effect of area on predictions changes as area varies.

Here's an ALE graph example:
![image](https://github.com/SamLB9/AI_Explainability/assets/106078401/bcf135b6-bf81-4afc-a616-a9744e5b6880)

Accumulated Local Effects provides a global view of the impact of a variable on predictions, taking into account possible interactions with other variables. This provides a better understanding of the relationship between the feature under study and the model's predictions on the data set.

### There are several popular Python libraries for studying the explicability and interpretability of AI models. Here are some of the main libraries used for this purpose:
1) SHAP (SHapley Additive exPlanations): SHAP is a very powerful and popular library for explaining ML model predictions. It relies on SHAP values to quantify the importance of each feature in a prediction. The library offers various tools and visualizations, including the SHAP Summary Plot, for understanding the impact of each variable on predictions.
2) LIME (Local Interpretable Model-agnostic Explanations): LIME is a library that provides local explanations for ML models of any type. It generates simple local models around a specific prediction to explain its behavior. LIME is often used to explain black-box models, such as deep neural networks.
3) ELI5 (Explain Like I'm 5): ELI5 is a library that offers explanations for various models, including models from scikit-learn, xgboost, and others. It provides a simple API for displaying weights, feature contributions, and activation maps.
4) Yellowbrick: Yellowbrick is a visualization library for the analysis and interpretation of ML models. It offers a variety of visualizations, including ROC curve graphs, heat maps, and feature contribution graphs.
5) AIX360 (AI Explainability 360): AIX360 is a library developed by IBM Research to study explainability in AI. It offers a variety of tools for understanding AI models, including techniques for explaining the predictions of computer vision and natural language processing models.
6) InterpretML: InterpretML is a Microsoft library for interpreting and understanding ML models, providing various techniques such as ALE graphs, ICE graphs and correlation maps.

These libraries offer different approaches and visualizations to help study and understand ML models. You can choose the library that best suits your needs and your specific model.

### In conclusion, these methods produce different explanations, but they seem to be consistent with each other, since certain variables are found in all the explanations. Finally, explainability methods vary in terms of techniques used, visualization and type of explanation. One of the difficulties therefore lies in evaluating these methods. However, it is possible to define criteria for comparing them, and to examine their respective advantages and disadvantages. These criteria can be used to select the right method(s) for a given use case.

## Which models are the most interpretable and explainable?

The most interpretable and explainable Machine Learning models are generally simple, linear models. Here are a few examples of models that are renowned for their interpretability and explicability:
- Linear regression: This is one of the simplest and most comprehensible models in ML. It models the linear relationship between input and output variables. Feature coefficients provide a direct indication of their importance in the model.
- Logistic regression: This model is used for binary classification. It is also relatively simple to interpret, as the coefficients can be associated with the contribution of each feature to the classification.
- Decision trees: Decision trees are easy to understand because they represent decisions in tree form. Each node in the tree represents a feature, and each branch corresponds to a binary decision based on that feature.
- Gradient Boosting Machines: Although more complex than simple decision trees, gradient boosting models retain a certain interpretability, especially when their depth is limited.
- Shallow neural networks: Shallow neural networks with a few hidden layers can be easier to interpret than deep networks. The low complexity makes it easier to understand how inputs are transformed into outputs.

In contrast, Machine Learning models that are generally less interpretable and explainable include:
- Deep neural networks: Deep neural networks, especially with many hidden layers, can be very complex, making it difficult to interpret their decisions. Internal mechanisms are often difficult to grasp.
- Support vector machines (SVMs) with complex kernels: SVMs can be interpretable with linear kernels, but when complex kernels are used, the relationship between features and predictions may be less clear.
- Recurrent neural networks (RNN) and Transformer-based natural language processing models: These models are designed for complex tasks such as text translation or text generation, and their sophisticated architectures make them difficult to interpret.

It's important to note that certain techniques, such as perturbation analysis, activation maps and other visualization methods, can help make certain aspects of the models more interpretable, even for more complex models. However, this will always depend on the level of complexity of the model and the specific task for which it is being used.

The most complex models, such as deep neural networks, are models that will never be interpretable, but may be explainable in the future, because today it's utopian to fully understand this kind of model.

Here's a graph that explains the complexity of the different models and their impact on interpretability:
![](https://github.com/SamLB9/AI_Explainability/blob/721bbfcc6f031b5c4e287755e2932ac1902c5d07/Graphes_ComparaisonModelsEvsI.png)

## Conclusion:
The interpretability and explicability of Machine Learning models are crucial research topics today. While advances in Artificial Intelligence have opened up new perspectives in fields such as medicine, finance and many others, it is becoming essential to understand how models make their decisions. However, despite efforts to make models more interpretable and explicable, the increasing complexity of ML architectures has made this task particularly challenging.

Simple, linear models, such as linear regressions and decision trees, have demonstrated their ability to be interpreted and explained with relative ease. Nevertheless, in fields such as medicine, where critical decisions have to be made, more sophisticated and powerful models, such as deep neural networks, are often required to achieve high performance. However, these complex models are often regarded as a "black box", as their inner workings are difficult to decipher.

The need to interpret and explain these complex models is more urgent than ever, especially when they are used to make decisions impacting the health, privacy or fundamental rights of individuals. Progress is being made with the use of interpretive methods, such as LIME and counterfactual explanations, but fully understanding all AI models remains a major challenge.

In this context, further research into the interpretability and explicability of ML models is essential. These efforts are needed to develop tools and approaches that will enable users and healthcare experts to better understand how models make decisions. While perfect understanding of all AI models may seem utopian, significant progress can be made by making models more transparent and providing clear and reliable explanations, to ensure responsible and ethical use of AI in fields as crucial as medicine and beyond.


### Bibliography:
- https://www.aqsone.com/blog/2022/data-science-fr/interpretabilite-et-explicabilite-des-modeles-de-machine-learning/
- https://www.kereval.com/explicabilite-de-lintelligence-artificielle/
- https://larevueia.fr/explicabilite-des-modeles-ne-croyez-pas-aveuglement-ce-que-lia-vous-dit/
- https://www.jmlr.org/papers/volume21/19-1035/19-1035.pdf
- https://www.nexialog.com/wp-content/uploads/2022/03/Interpretabilite-du-ML-Nexialog-Consulting.pdf




