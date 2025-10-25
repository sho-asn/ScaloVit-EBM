import numpy as np

def detect_with_ema(energies, alpha=0.05, threshold=1.0):
    """
    Detects anomalies using an Exponential Moving Average (EMA) of energy scores.

    An alarm is triggered if the EMA exceeds a given threshold.

    Args:
        energies (list or np.ndarray): A sequence of per-chunk energy scores from the model.
        alpha (float): The smoothing factor for the EMA, between 0 and 1. 
                       A smaller alpha makes it more sensitive to slow, sustained changes.
        threshold (float): The threshold to trigger an alarm.

    Returns:
        list[bool]: A list of booleans, where True indicates an alarm at that time step.
        list[float]: The calculated EMA value at each time step.
    """
    if len(energies) == 0:
        return [], []

    ema_values = []
    alarms = []
    
    # Initialize the first EMA value
    ema = energies[0]
    ema_values.append(ema)
    alarms.append(ema > threshold)

    # Calculate EMA for the rest of the energies
    for e in energies[1:]:
        ema = alpha * e + (1 - alpha) * ema
        ema_values.append(ema)
        alarms.append(ema > threshold)
        
    return alarms, ema_values


def detect_with_cusum(energies, baseline=0.5, h=5.0, k=0.1):
    """
    Detects anomalies using the Cumulative Sum (CUSUM) algorithm.

    This method is very effective at detecting small but persistent shifts in the mean
    of a signal, making it ideal for gradual drift detection.

    Args:
        energies (list or np.ndarray): A sequence of per-chunk energy scores from the model.
        baseline (float): The expected mean of the energy scores under normal conditions.
                          This should be tuned on a long run of normal data (e.g., median).
        h (float): The detection threshold. An alarm is raised when the cumulative sum exceeds h.
        k (float): A slack parameter, representing an allowable deviation from the baseline.
                   Typically a small fraction of the standard deviation of normal energies.

    Returns:
        list[bool]: A list of booleans, where True indicates an alarm at that time step.
        list[float]: The calculated cumulative sum value (s) at each time step.
    """
    if len(energies) == 0:
        return [], []

    s = 0
    cusum_values = []
    alarms = []

    for e in energies:
        # The core CUSUM update rule: accumulate deviations above the baseline+slack
        s = max(0, s + (e - baseline - k))
        cusum_values.append(s)
        alarms.append(s > h)
        
    return alarms, cusum_values


if __name__ == '__main__':
    # --- Example Usage ---

    # A sequence of energy scores simulating a gradual drift
    # Starts normal, slowly drifts up, then has a larger anomaly
    normal_energies = np.random.normal(0.5, 0.2, 50).tolist()
    drift_energies = np.linspace(0.8, 2.0, 30).tolist()
    anomaly_energies = [2.5, 3.0, 1.5, 4.0, 3.5]
    test_energies = normal_energies + drift_energies + anomaly_energies

    print(f"Total number of energy scores: {len(test_energies)}\n")

    # --- 1. Using EMA ---
    # A low alpha (0.1) makes it sensitive to trends
    # The threshold needs to be tuned based on expected EMA values for normal data
    ema_alarms, ema_values = detect_with_ema(test_energies, alpha=0.1, threshold=1.5)
    print("---" + "EMA Detection Results" + "---")
    print(f"EMA Alarms triggered: {sum(ema_alarms)} times")
    # Find the first alarm index
    try:
        first_ema_alarm = ema_alarms.index(True)
        print(f"First EMA alarm at time step: {first_ema_alarm}")
    except ValueError:
        print("No EMA alarms triggered.")
    print("---------------------------\n")


    # --- 2. Using CUSUM ---
    # The baseline, h (threshold), and k (slack) must be tuned on real normal data.
    # Here we assume a baseline energy of 0.5 for normal data.
    cusum_alarms, cusum_values = detect_with_cusum(test_energies, baseline=0.5, h=4.0, k=0.2)
    print("---" + "CUSUM Detection Results" + "---")
    print(f"CUSUM Alarms triggered: {sum(cusum_alarms)} times")
    # Find the first alarm index
    try:
        first_cusum_alarm = cusum_alarms.index(True)
        print(f"First CUSUM alarm at time step: {first_cusum_alarm}")
    except ValueError:
        print("No CUSUM alarms triggered.")
    print("----------------------------\n")

    # You can use a plotting library like matplotlib to visualize the results:
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(15, 6))
    # plt.plot(test_energies, label='Raw Energy', alpha=0.6)
    # plt.plot(ema_values, label='EMA of Energy', linestyle='--')
    # plt.plot(cusum_values, label='CUSUM Value', linestyle=':')
    # plt.axhline(y=1.5, color='red', linestyle='--', label='EMA Threshold')
    # plt.axhline(y=4.0, color='green', linestyle=':', label='CUSUM Threshold')
    # plt.legend()
    # plt.title('Anomaly Scoring Example')
    # plt.xlabel('Time Step')
    # plt.ylabel('Value')
    # plt.show()
