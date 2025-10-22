# Train-Test Split Guide

## Overview

This guide explains how to properly validate your face recognition model using the **80/20 train-test split** methodology.

---

## 🎯 **What is Train-Test Split?**

### **The Concept:**

When you collect 30 samples per person:
- **80% (24 images)** → Used for training the model
- **20% (6 images)** → Held out for testing (unseen data)

This validates that your model can recognize people from **new photos**, not just memorize training images.

### **Why It's Important:**

**Without Split:**
- Train on all 30 images
- Test on same 30 images
- Model might just memorize → 100% accuracy (misleading!)
- Real-world performance: Could be only 70%

**With Split:**
- Train on 24 images
- Test on 6 NEW images
- Realistic accuracy measurement
- Confidence in real-world performance

---

## 🛠️ **How to Use**

### **Method 1: Automatic Split (Recommended)** ⭐

Use the new training script that automatically splits data:

```bash
# Capture 30 samples per person (as usual)
python examples/capture_training_data.py --name "Person 1" --samples 30
python examples/capture_training_data.py --name "Person 2" --samples 30
python examples/capture_training_data.py --name "Person 3" --samples 30

# Train with automatic 80/20 split
python examples/train_with_validation.py --data training_data --output models/recognizer.yml

# The script will:
# 1. Automatically split each person's data (80% train, 20% test)
# 2. Train on 80%
# 3. Validate on 20%
# 4. Show accuracy metrics
# 5. Save model if accuracy is good
```

### **Example Output:**

```
📂 Found 3 person(s) in training data

👤 Person 1:
   Total: 30 images
   Train: 24 images (ID: 0)
   Test:  6 images

👤 Person 2:
   Total: 30 images
   Train: 24 images (ID: 1)
   Test:  6 images

👤 Person 3:
   Total: 30 images
   Train: 24 images (ID: 2)
   Test:  6 images

📊 Dataset Summary:
   Training samples: 72
   Test samples: 18
   Total people: 3

🎓 Training model on 72 samples...
✅ Training completed!

EVALUATING MODEL ON TEST SET
============================================================
✓ Test 1/18: True=Person 1, Pred=Person 1, Conf=42.3
✓ Test 2/18: True=Person 1, Pred=Person 1, Conf=38.7
✓ Test 3/18: True=Person 1, Pred=Person 1, Conf=45.1
...

TEST SET RESULTS
============================================================
Total Test Samples:     18
Correct Predictions:    17
Incorrect Predictions:  1

Accuracy:               94.44% (goal: ≥90%)
False Acceptance Rate:  5.56% (goal: ≤2%)
False Rejection Rate:   0.00% (goal: ≤5%%)

✅ MODEL PASSED VALIDATION!
```

### **Custom Split Ratio:**

```bash
# 70/30 split (30% for testing)
python examples/train_with_validation.py --data training_data --output models/recognizer.yml --split 0.3

# 90/10 split (10% for testing)
python examples/train_with_validation.py --data training_data --output models/recognizer.yml --split 0.1
```

---

## 📊 **Understanding the Results**

### **Key Metrics Explained:**

**1. Accuracy**
- **Formula:** (Correct Predictions / Total Tests) × 100%
- **Target:** ≥ 90%
- **Meaning:** How often the model correctly identifies people

**2. False Acceptance Rate (FAR)**
- **Formula:** (Wrong Person Identified / Total Tests) × 100%
- **Target:** ≤ 2%
- **Meaning:** How often it mistakes someone for another person

**3. False Rejection Rate (FRR)**
- **Formula:** (Failed to Recognize / Total Tests) × 100%
- **Target:** ≤ 5%
- **Meaning:** How often it fails to recognize an enrolled person

**4. Per-Person Accuracy**
- Shows accuracy for each individual
- Identifies if specific people need more training data

### **Example Interpretation:**

```
Accuracy: 92.5%  → ✅ Excellent! Above 90% target
FAR: 1.8%        → ✅ Great! Below 2% target  
FRR: 5.7%        → ⚠️  Slightly high, acceptable but could improve

Per-Person:
  Person 1: 95%  → ✅ Excellent
  Person 2: 88%  → ⚠️  Needs more training data
  Person 3: 94%  → ✅ Excellent
```

**Action:** Capture more samples for Person 2.

---

## 🔄 **Comparison: With vs Without Split**

### **Scenario: 3 People, 30 Samples Each**

**Without Validation (Old Method):**
```bash
python examples/train_recognizer.py --data training_data --output models/recognizer.yml

Output:
✅ Training completed!
   • Samples processed: 90
   • Unique labels: 3

# BUT: No idea if it will work on new photos!
# Could be overfitting to training images
```

**With Validation (New Method):**
```bash
python examples/train_with_validation.py --data training_data --output models/recognizer.yml

Output:
✅ Training completed!
   • Training samples: 72 (80%)
   • Test samples: 18 (20%)
   
TEST RESULTS:
   • Accuracy: 94.4%
   • FAR: 5.6%
   • FRR: 0.0%

✅ MODEL PASSED VALIDATION!

# NOW you know it works on unseen photos!
```

---

## 📈 **Best Practices**

### **1. Recommended Sample Counts**

| People | Samples per Person | Training | Testing | Total |
|--------|-------------------|----------|---------|-------|
| 1-10   | 30                | 24       | 6       | 240-300 |
| 10-50  | 25                | 20       | 5       | 250-1,250 |
| 50+    | 20                | 16       | 4       | 1,000+ |

### **2. When to Increase Samples**

**Increase from 30 to 40-50 if:**
- Accuracy < 90%
- High FAR or FRR
- Challenging conditions (varied lighting, poses)
- High-security requirements

### **3. When Split Ratio Matters**

**Use 80/20 (default) when:**
- ✅ Standard validation
- ✅ 25-30 samples per person
- ✅ General attendance system

**Use 70/30 when:**
- More rigorous testing needed
- 30+ samples per person

**Use 90/10 when:**
- Limited data (< 20 samples per person)
- Need more training data

---

## 🎯 **Workflow Comparison**

### **Complete Workflow with Validation:**

```bash
# Step 1: Capture data (30 samples per person)
python examples/capture_training_data.py --name "Alice" --samples 30
python examples/capture_training_data.py --name "Bob" --samples 30
python examples/capture_training_data.py --name "Charlie" --samples 30

# Step 2: Train with validation (automatic 80/20 split)
python examples/train_with_validation.py --data training_data --output models/recognizer.yml

# Step 3: Check results
# ✅ Accuracy ≥ 90%? → Deploy!
# ⚠️  Accuracy < 90%? → Capture more data for low-scoring people

# Step 4: Deploy
python examples/live_recognition_db.py --model models/recognizer.yml
```

---

## 🔍 **Troubleshooting**

### **Issue: Accuracy Below 90%**

**Possible Causes:**
1. Not enough training data per person
2. Poor quality images (blurry, poor lighting)
3. Inconsistent capture conditions
4. Too similar-looking people

**Solutions:**
```bash
# 1. Capture more samples (40-50)
python examples/capture_training_data.py --name "Person" --samples 50

# 2. Retrain with lower threshold (stricter)
python examples/train_with_validation.py --data training_data --output models/recognizer.yml --threshold 45

# 3. Check per-person accuracy and focus on low scorers
```

### **Issue: High False Acceptance Rate**

**Meaning:** Model confuses different people

**Solutions:**
- Lower threshold (stricter matching)
- Capture more diverse samples per person
- Ensure people look sufficiently different

```bash
# Lower threshold to 40 (stricter)
python examples/train_with_validation.py --data training_data --output models/recognizer.yml --threshold 40
```

### **Issue: High False Rejection Rate**

**Meaning:** Model fails to recognize enrolled people

**Solutions:**
- Higher threshold (more permissive)
- More training samples per person
- Better quality training images

```bash
# Higher threshold to 60 (more permissive)
python examples/train_with_validation.py --data training_data --output models/recognizer.yml --threshold 60
```

---

## 📊 **K-Fold Cross-Validation (Advanced)**

For more rigorous validation with limited data:

```python
# Advanced: 5-fold cross-validation
# Splits data into 5 parts, trains 5 times
# Each time uses different part for testing
# Average accuracy across all 5 runs

# This is more complex and usually unnecessary
# 80/20 split is sufficient for most cases
```

---

## ✅ **Summary**

### **Why Use Train-Test Split?**

1. ✅ **Validates real-world performance**
2. ✅ **Detects overfitting**
3. ✅ **Provides confidence metrics**
4. ✅ **Identifies weak individuals**
5. ✅ **Professional ML practice**

### **Quick Commands:**

```bash
# Capture data
python examples/capture_training_data.py --name "Name" --samples 30

# Train with validation
python examples/train_with_validation.py --data training_data --output models/recognizer.yml

# Deploy if accuracy ≥ 90%
python examples/live_recognition_db.py --model models/recognizer.yml
```

### **Success Criteria:**

✅ Accuracy ≥ 90%  
✅ FAR ≤ 2%  
✅ FRR ≤ 5%  
✅ All individuals ≥ 85% accuracy  

**If met → Model is ready for production!** 🎉

---

## 🎓 **Further Reading**

- **Cross-validation:** More rigorous testing (usually unnecessary for attendance)
- **Stratified sampling:** Ensures balanced split (automatic in our implementation)
- **Holdout validation:** Our approach (simple, effective)
- **Confusion matrix:** Detailed error analysis (advanced)

**For attendance systems, 80/20 split is the sweet spot!** ✨

