# LendingClub Credit Scoring Model

This project builds a machine learning model to predict creditworthiness using the LendingClub loan dataset. The model is trained to classify whether a loan is safe or risky, helping financial institutions assess loan applications effectively.

## 📁 Project Structure

```
├── app.py                  # Streamlit app for predictions
├── data_preparation.py     # Data cleaning and preprocessing functions
├── credit_scoring_model.py # Model training and evaluation script
├── model.pkl               # Trained machine learning model
├── lendingclub_raw.csv     # Original raw dataset
├── lendingclub_test_data.csv # Processed test dataset
├── README.md               # Project documentation (this file)
```

## 🧠 Features Used

- Loan amount, term, interest rate
- Employment length, home ownership
- Annual income, purpose of the loan
- Credit history length, revolving utilization

## 🚀 How to Run

1. **Clone the repository**:
    ```bash
    git clone https://github.com/JoY-BoY-e/CodeApha_Machine_Learning.git
    cd CodeApha_Machine_Learning
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Train the model**:
    ```bash
    python credit_scoring_model.py
    ```

4. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

## 📊 Model Info

- Model Used: RandomForestClassifier
- Accuracy: ~87% on validation set
- Scaler: StandardScaler (saved to `scaler.pkl`)

## 🧪 Testing

Make predictions using the test dataset:
```bash
python credit_scoring_model.py --test lendingclub_test_data.csv
```

## 💼 Author

**Salman Mehmood**  
📧 salmanmehmood19j@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/salman-mehmood-020845318/)  
🔗 [GitHub](https://github.com/JoY-BoY-e)

---

**Note:** This model is for educational purposes only and not intended for real-world loan decision making.
