# Logistic regression with selected variables
X_logit = participant_data[selected_behaviors]
y_logit = participant_data['high_fear']  # 4=high fear, 1-3=not high fear

# Statsmodels logistic regression (provides detailed statistics)
X_logit_sm = sm.add_constant(X_logit)
logit_model = sm.Logit(y_logit, X_logit_sm).fit(disp=0)  # disp=0 suppresses iteration output
print("\nLogistic Regression Results (statsmodels):")
print(logit_model.summary())

# Scikit-learn logistic regression (for prediction and ROC curve)
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_logit, y_logit)

# Influence of each behavior on high fear prediction
feature_importance = pd.DataFrame({
    'Behavior': selected_behaviors,
    'Coefficient': log_reg.coef_[0],
    'Odds Ratio (OR)': np.exp(log_reg.coef_[0])
})
feature_importance = feature_importance.sort_values('Odds Ratio (OR)', ascending=False)
print("\nLogistic Regression: Behavior influence on high fear:")
print(feature_importance)

# ROC curve
y_pred_prob = log_reg.predict_proba(X_logit)[:, 1]
fpr, tpr, _ = roc_curve(y_logit, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression: ROC Curve for Predicting High Fear')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('logistic_regression_roc.png')
