## Data Leak couses
- The duplicates in the dataset
- The data that will not be available at the time of prediction aka monthly_payment_capacity
- The last_audit_team_id was really correlated with the target variable and did not make any sense to be there
Removed the things listed above from the dataset to avoid data leak