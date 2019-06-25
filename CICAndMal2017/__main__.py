import CICAndMal2017.adware_2_model_tuning as adware
import CICAndMal2017.ransomware_2_model_tuning as ransomware
import CICAndMal2017.scareware_2_model_tuning as scareware
import CICAndMal2017.sms_malware_2_model_tuning as sms_malware
import time
if __name__ == '__main__':
    start_time = time.time()

    print("Performing CICAndMal2017 attacks model tuning performance scoring")
    print("Model Chosen: Random Forest")

    print("___________________ADWARE PERFORMANCE___________________")
    adware.calc()
    print("___________________RANSOMWARE PERFORMANCE___________________")
    ransomware.calc()
    print("___________________SCAREWARE PERFORMANCE___________________")
    scareware.calc()
    print("___________________SMS MALWARE PERFORMANCE___________________")
    sms_malware.calc()

    end_time = time.time()

    execution_time = end_time - start_time

    if execution_time < 60:
        print("Perfromance scoring completed in {} seconds".format(execution_time))
    else:
        print("Perfromance scoring completed in {}.{} minutes".format(execution_time / 60, execution_time % 60))
