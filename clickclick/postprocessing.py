import copy
import numpy as np
import pandas as pd


LABEL_MAP = {
    'Motorola-Droid-Maxx': 0,
    'Motorola-Nexus-6': 1,
    'Samsung-Galaxy-Note3': 2,
    'Motorola-X': 3,
    'iPhone-4s': 4,
    'iPhone-6': 5,
    'Samsung-Galaxy-S4': 6,
    'Sony-NEX-7': 7,
    'LG-Nexus-5x': 8,
    'HTC-1-M7': 9,
}


def generate_submission(pred_list, filenames, output_path, n_tta=11):
    """Calculates the best predictions for each CV models output, computes the
    average predicted probability across images, then saves the final results
    to `output_path` ready for evaluation
    
    Params:
        pred_list (list): 
            list storing the predictions from each CV model 
        filenames (list): 
            list storing the filenames used to predict `pred_list`
        output_path (str): 
            name of the path for storing a .csv of the final prediction for 
            each image in `filenames`
        n_tta (int): 
            number of TTA's used for prediction. Default: 11
    """
    k = len(pred_list)
    cv_pred_list = []
    for cv_idx in range(k):
        raw_cv_preds = pred_list[cv_idx]
        tta_preds = []
        
        for i in range(n_tta):
            tta_preds.append(raw_cv_preds[i::n_tta])
        cv_preds = calculate_cv_preds(tta_preds)
        cv_pred_list.append(cv_preds)

    final_probs = np.mean(cv_pred_list, axis=0)
    final_preds = final_probs.argmax(axis=1)
    
    rev_label_dict = {v: k for k, v in LABEL_MAP.items()}
    label_preds = [rev_label_dict[lbl] for lbl in final_preds]
    data = list(zip(filenames, label_preds))
    
    df_sub = pd.DataFrame(data, columns=['fname', 'camera'])
    df_sub.to_csv(output_path, index=False)
    print("Saved predictions to {}".format(output_path))


def calculate_cv_preds(tta_preds):
    """Calculates the final predicted probabilities given the raw predictions from
    a CV model
    
    Params:
        tta_preds (list): 
            list with length `n_tta` where each entry is a numpy array of shape 
            (n_samples, n_classes) storing each images prediction for a TTA input
    Returns:
        pred_probs (numpy array): 
            array of shape (n_samples, n_classes) storing each images predicted 
            probabilities
    """
    mean_preds = geometric_mean(tta_preds)
    coefs = find_best_coefs(mean_preds)
    pred_probs = calculate_pred_probs(mean_preds, coefs)
    return pred_probs


def geometric_mean(tta_pred_list):
    """Computes the geometric mean across each TTA prediction for every image
        
    params:
        tta_pred_list (list): 
            list with length `n_tta` where each entry is a numpy array storing 
            each each images prediction for a TTA input
    returns:
        result (numpy array): 
            array of shape (n_samples, n_classes) with each images geometric 
            mean across their TTA predictions 
    
    """
    tta_pred_list = tta_pred_list.copy()
    result = np.ones_like(tta_pred_list[0])
    for tta_pred in tta_pred_list:
        result *= tta_pred
    result **= 1. / len(tta_pred_list)
    return result


def find_best_coefs(preds, alpha=1e-3, iterations=10000):
    """Returns the class coefficients that return the best distribution
    over the predictions
    
    Params:
        preds (numpy array): 
            array with shape (n_samples,n_classes) with each images 
            original mean prediction
        alpha (float): 
            scalar to penalize the coefficient for the class with the 
            largest predicted distribution. Default: 0.001
        iterations (int): 
            number of iterations to reduce the largest class coefficient 
            by `alpha` and recompute the new `coefs` score. 
            Default: 10000
    Returns:
        best_coefs (list): 
            list storing the class coefficients that returned the best 
            score
            
    """
    preds = copy.deepcopy(preds)
    coefs = [1 for _ in range(preds.shape[1])]
    best_coefs = coefs.copy()
    best_score = compute_coefs_score(preds, coefs)

    for i in range(iterations):
        dist = get_labels_distribution(preds, coefs)
        label = np.argmax(dist)
        coefs[label] -= alpha
        score = compute_coefs_score(preds, coefs)
        if score > best_score:
            best_score = score
            best_coefs = coefs.copy()
    return best_coefs


def compute_coefs_score(preds, coefs):
    """Scores how evenly distributed a set of predictions are given `preds` 
    and `coefs` 
    
    Params:
        preds (numpy array): 
            array with shape (n_samples,n_classes) with each images original 
            mean prediction
        coefs (list): 
            list storing each classes weight
    Returns:
        score (float): 
            score for the final predicted output used to validate the 
            performance of `coefs` passed in
    """
    counter = get_labels_distribution(preds, coefs)
    score = 0.
    n_classes = len(coefs)
    max_class_score = n_classes * 1e-2
    # The maximum score for each class is `n_classes` / 100
    # The maximum score total score is 1.
    for label in range(n_classes):
        score += min(counter[label] / len(preds), 
                     max_class_score)
    return score


def get_labels_distribution(preds, coefs):
    """Counts the number of predictions for each image with a set of coefficients
    
    Params:
        preds (numpy array): 
            array with shape (n_samples,n_classes) with each images original 
            mean prediction
        coefs (list): 
            list storing each classes weight        
    Returns:
        counter (dict): 
            dictionary with each class and it's number of samples with the class 
            the prediction (highest probabiity)
    """
    pred_probs = calculate_pred_probs(preds, coefs)
    final_preds = preds.argmax(axis=1)
    counter = {lbl: 0 for lbl in range(len(LABEL_MAP))}
    for lbl in final_preds:
        counter[lbl] = counter.get(lbl) + 1
    return counter


def calculate_pred_probs(preds, coefs):
    """Calculates the final predicted probability for each image
    
    Params:
        preds (numpy array): 
            array with shape (n_samples,n_classes) with each images original
            mean prediction
        coefs (list): 
            list storing each classes weight
    Returns:
        final_preds (numpy array): 
            array with shape (n_samples,n_classes) with every images predicted 
            probabilities after multiplying each class prediction with the 
            corresponding coefficient 
        
    """
    pred_probs = copy.deepcopy(preds)
    for i in range(len(coefs)): # 10
        pred_probs[:,i] *= coefs[i]
    return pred_probs