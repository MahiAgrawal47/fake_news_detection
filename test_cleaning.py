# import pickle
# import numpy as np

# vectorizer = pickle.load(open('artifacts/vectorizer.pkl','rb'))
# model = pickle.load(open('artifacts/best_model.pkl','rb'))

# feature_names = vectorizer.get_feature_names_out()
# coefs = model.coef_[0]

# top_real = np.argsort(coefs)[-25:]
# top_fake = np.argsort(coefs)[:25]

# print("Top REAL indicators:")
# for i in top_real:
#     print(f"  {feature_names[i]}: {coefs[i]:.3f}")

# print("\nTop FAKE indicators:")
# for i in top_fake:
#     print(f"  {feature_names[i]}: {coefs[i]:.3f}")

from src.components.predict import prediction
print(prediction("Scientists announced a breakthrough in battery technology yesterday, claiming it could double electric vehicle range."))
print(prediction("BREAKING: Aliens land in New York, government covers up the truth, sources say!!!"))