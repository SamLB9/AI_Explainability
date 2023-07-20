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

### 1) LIME (Local Interpretable Model-agnostic Explanations):

One of the most popular explicability methods is LIME (Local Interpretable Model-agnostic Explanations). Its principle is simple: locally approximate the complex model with a simpler, and therefore interpretable, model. This method makes it possible to explain the model's decision concerning a particular observation. New instances close to the latter are generated by perturbing the values of the variables. These new instances are weighted according to their proximity to the instance to be explained. Predictions are then made for these new instances. A simple model, such as a linear regression, is finally fitted to these new instances and the associated predictions to produce the explanation.

Example of a local explanation generated by the LIME method:

![image](https://github.com/SamLB9/AI_Explainability/assets/106078401/2ce4617f-8b1e-4931-9147-d1a20730348d)

For the chosen customer, the explanation provided by the LIME method indicates that the probability of cancellation is 83%, and that this prediction is positively influenced by the fact that this customer has taken out a monthly contract with optical fiber.

### 2) Ancres:

The Anchors method has been developed to overcome some of LIME's problems, notably concerning the generalizability of explanations, which is not clearly defined. The principle of this second method is to define decision rules that anchor a prediction, using an optimized search algorithm.

Example of a local explanation generated by the ancres method:

![image](https://github.com/SamLB9/AI_Explainability/assets/106078401/62ac21db-a884-48ee-a69e-b55c9fe9a5c8)

The explanation given for the chosen customer indicates the variables that anchor the churn prediction: a rather low seniority (less than 29 months), a type of electronic payment, a monthly contract, etc. The added value of this method is linked to an additional piece of information, the coverage. In this case, 8% of customers have the characteristics defined by the anchor.

### 3) Counterfactual explanations:

In contrast to the previous method, which focuses on the variables that anchor predictions, the counterfactual explanation method looks for those whose change modifies the prediction. Several approaches exist to find the smallest possible change that modifies the prediction, including the naive trial-and-error approach or the use of an optimization algorithm. The advantage of this method is that it generates several explanations for a single decision.

Example of a counterfactual explanation:

![image](https://github.com/SamLB9/AI_Explainability/assets/106078401/9f34fe28-4f5a-49e3-b150-e9b75c86ae1a)

Still on the subject of the chosen customer, a first counterfactual explanation consists in the modification of his type of contract and his type of payment; with these two changes, the model predicts non-cancellation. The same applies if only the customer's seniority is modified, from 13 to 65 months.

### 4) Individual Conditional Expectation(ICE):
Individual Conditional Expectation (ICE) is a technique used to understand how a particular variable affects the predictions of a Machine Learning model. Essentially, it involves decomposing the effect of that variable on predictions, by showing how the values of that variable individually influence the results of the model.

To better understand, imagine you have a prediction model, for example one that predicts the price of a house based on its characteristics such as surface area, number of bedrooms, etc. You want to understand how the surface area of the house affects the price of the house. You want to understand how the size of the house affects price predictions.

With Individual Conditional Expectation, you'll take the following steps:
1) Fix all other variables: For each data instance (i.e. a specific house in our example), keep the other characteristics fixed, such as number of bedrooms, location, etc.
2) Vary the area: You will now vary only the area of the house for this data instance, taking different possible area values.
3) Observe predictions: At each area value, make predictions with the model and record the results (i.e. the predicted prices). You will obtain a series of predictions corresponding to different area values.
4) Plot the ICE graph: Finally, plot the ICE graph, which shows on the x-axis the different area values you've tested, and on the y-axis, the corresponding predictions. The graph will show you how the predictions change as the area of the house varies.

ICE allows you to visualize how each individual variable affects the model's predictions, regardless of interactions with other variables. This can provide valuable information on the relationship between features and model predictions, and enable a better understanding of the factors influencing prediction results.

### In conclusion, these three methods produce different explanations, but they seem to be consistent with each other, since certain variables are found in all the explanations. This is the case for contract type, which appears to be an important factor in prediction. This seems relevant, since it is generally easier to cancel a monthly contract, often without commitment. Finally, explicability methods vary in terms of the techniques used, visualization and type of explanation. One of the difficulties therefore lies in evaluating these methods. However, it is possible to define criteria for comparing them, and to look at their respective advantages and disadvantages. These criteria can be used to select the right method(s) for a given use case.

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


Continuer la recherche sur ce sujet:
'Si vous êtes intéressé par ce sujet, nous vous invitons à vous documenter sur d’autres méthodes d’interprétabilité des modèles de machine learning telles que : les techniques reposant sur l’analyse de « graphiques ICE et PDP », les méthodes dites de « Permutation Feature Importance », les « explications contrefactuelles » ou encore les « ancres ».'



