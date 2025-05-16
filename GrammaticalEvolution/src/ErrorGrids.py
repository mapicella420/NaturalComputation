'''
CLARKE ERROR GRID ANALYSIS      ClarkeErrorGrid.py

Need Matplotlib Pyplot


The Clarke Error Grid shows the differences between a blood glucose predictive measurement and a reference measurement,
and it shows the clinical significance of the differences between these values.
The x-axis corresponds to the reference value and the y-axis corresponds to the prediction.
The diagonal line shows the prediction value is the exact same as the reference value.
This grid is split into five zones. Zone A is defined as clinical accuracy while
zones C, D, and E are considered clinical error.

Zone A: Clinically Accurate
    This zone holds the values that differ from the reference values no more than 20 percent
    or the values in the hypoglycemic range (<70 mg/dl).
    According to the literature, values in zone A are considered clinically accurate.
    These values would lead to clinically correct treatment decisions.

Zone B: Clinically Acceptable
    This zone holds values that differe more than 20 percent but would lead to
    benign or no treatment based on assumptions.

Zone C: Overcorrecting
    This zone leads to overcorrecting acceptable BG levels.

Zone D: Failure to Detect
    This zone leads to failure to detect and treat errors in BG levels.
    The actual BG levels are outside of the acceptable levels while the predictions
    lie within the acceptable range

Zone E: Erroneous treatment
    This zone leads to erroneous treatment because prediction values are opposite to
    actual BG levels, and treatment would be opposite to what is recommended.


SYNTAX:
        plot, zone = clarke_error_grid(ref_values, pred_values, title_string)

INPUT:
        ref_values          List of n reference values.
        pred_values         List of n prediciton values.
        title_string        String of the title.

OUTPUT:
        plot                The Clarke Error Grid Plot returned by the function.
                            Use this with plot.show()
        zone                List of values in each zone.
                            0=A, 1=B, 2=C, 3=D, 4=E

EXAMPLE:
        plot, zone = clarke_error_grid(ref_values, pred_values, "00897741 Linear Regression")
        plot.show()

References:
[1]     Clarke, WL. (2005). "The Original Clarke Error Grid Analysis (EGA)."
        Diabetes Technology and Therapeutics 7(5), pp. 776-779.
[2]     Maran, A. et al. (2002). "Continuous Subcutaneous Glucose Monitoring in Diabetic
        Patients" Diabetes Care, 25(2).
[3]     Kovatchev, B.P. et al. (2004). "Evaluating the Accuracy of Continuous Glucose-
        Monitoring Sensors" Diabetes Care, 27(8).
[4]     Guevara, E. and Gonzalez, F. J. (2008). Prediction of Glucose Concentration by
        Impedance Phase Measurements, in MEDICAL PHYSICS: Tenth Mexican
        Symposium on Medical Physics, Mexico City, Mexico, vol. 1032, pp.
        259261.
[5]     Guevara, E. and Gonzalez, F. J. (2010). Joint optical-electrical technique for
        noninvasive glucose monitoring, REVISTA MEXICANA DE FISICA, vol. 56,
        no. 5, pp. 430434.


Made by:
Trevor Tsue
7/18/17

Based on the Matlab Clarke Error Grid Analysis File Version 1.2 by:
Edgar Guevara Codina
codina@REMOVETHIScactus.iico.uaslp.mx
March 29 2013
'''



import matplotlib.pyplot as plt
import numpy as np


def above_line(x_1, y_1, x_2, y_2, act, pred, strict=False):
    if x_1 == x_2:
        return False
    y_line = ((y_1 - y_2) * act + y_2 * x_1 - y_1 * x_2) / (x_1 - x_2)
    return pred > y_line if strict else pred >= y_line


def below_line(x_1, y_1, x_2, y_2, act, pred, strict=False):
    return not above_line(x_1, y_1, x_2, y_2, act, pred, not strict)

def parkes_error_grid(ref_values, pred_values, title_string):
    """
    This function outputs the Parkes Error Grid region (encoded as integer)
    for a combination of actual and predicted value for type 1 diabetic patients
    Based on the article 'Technical Aspects of the Parkes Error Grid':
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3876371/
    """
    # Clear plot
    plt.clf()

    # Set up plot
    plt.scatter(ref_values, pred_values, marker='o', color='blue', s=3)
    plt.title(title_string + " Parkes Error Grid")
    plt.xlabel("Reference Concentration [mg/dL]")
    plt.ylabel("Prediction Concentration [mg/dL]")
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550])
    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550])
    plt.gca().set_facecolor('white')

    # Set axes lengths
    plt.gca().set_xlim([0, 550])
    plt.gca().set_ylim([0, 550])
    plt.gca().set_aspect((550) / (550))

    # Plot zone lines
    plt.plot([0, 550], [0, 550], ':', c='black')  # Theoretical 45 regression line
    plt.plot([50, 50, 170, 385, 550], [0, 30, 145, 300, 450], '-', c='black')
    plt.plot([0, 30, 140, 280, 430], [50, 50, 170, 380, 550], '-', c='black')
    plt.plot([120, 120, 260, 550], [0, 30, 130, 250], '-', c='black')
    plt.plot([0, 30, 50, 70, 260], [60, 60, 80, 110, 550], '-', c='black')
    plt.plot([250, 250, 550], [0, 40, 150], '-', c='black')
    plt.plot([0, 25, 50, 80, 125], [100, 100, 125, 215, 550], '-', c='black')
    plt.plot([0, 35, 50], [150, 155, 550], '-', c='black')

    # Add zone titles
    plt.text(20, 500, "E", fontsize=15);
    plt.text(75, 480, "D", fontsize=15);
    plt.text(150, 460, "C", fontsize=15);
    plt.text(250, 440, "B", fontsize=15);
    plt.text(300, 375, "A", fontsize=15);
    plt.text(350, 320, "A", fontsize=15);
    plt.text(400, 220, "B", fontsize=15);
    plt.text(430, 150, "C", fontsize=15);
    plt.text(475, 75, "D", fontsize=15);

    # Add zone titles


    rmse = 0
    result = np.all(pred_values == pred_values[0]) or np.isscalar(pred_values)
    if result:
        rmse = 1.0e+100
    else:
        # Statistics from the data
        zone = [0] * 5
        for i in range(len(ref_values)):
            # Zone E
            if above_line(0, 150, 35, 155, ref_values[i], pred_values[i]) and above_line(35, 155, 50, 550, ref_values[i], pred_values[i]):
                zone[4] += 1
            # Zone D - left upper
            elif (pred_values[i] > 100 and above_line(25, 100, 50, 125, ref_values[i], pred_values[i]) and
                    above_line(50, 125, 80, 215, ref_values[i], pred_values[i]) and above_line(80, 215, 125, 550, ref_values[i], pred_values[i])):
                zone[3] += 1
            # Zone D - right lower
            elif (ref_values[i] > 250 and below_line(250, 40, 550, 150, ref_values[i], pred_values[i])):
                zone[3] += 1
            # Zone C - left upper
            elif (pred_values[i] > 60 and above_line(30, 60, 50, 80, ref_values[i], pred_values[i]) and
                    above_line(50, 80, 70, 110, ref_values[i], pred_values[i]) and above_line(70, 110, 260, 550, ref_values[i], pred_values[i])):
                zone[2] += 1
            # Zone C - right lower
            elif (ref_values[i] > 120 and below_line(120, 30, 260, 130, ref_values[i], pred_values[i]) and below_line(260, 130, 550, 250, ref_values[i], pred_values[i])):
                rmse += 64 * np.abs(pred_values[i] - ref_values[i])
                zone[2] += 1
            # Zone B - left upper
            elif (pred_values[i] > 50 and above_line(30, 50, 140, 170, ref_values[i], pred_values[i]) and
                    above_line(140, 170, 280, 380, ref_values[i], pred_values[i]) and (ref_values[i] < 280 or above_line(280, 380, 430, 550, ref_values[i], pred_values[i]))):
                zone[1] += 1
            # Zone B - right lower
            elif (ref_values[i] > 50 and below_line(50, 30, 170, 145, ref_values[i], pred_values[i]) and
                    below_line(170, 145, 385, 300, ref_values[i], pred_values[i]) and (ref_values[i] < 385 or below_line(385, 300, 550, 450, ref_values[i], pred_values[i]))):
                zone[1] += 1
            else:
                # Zone A
                zone[0] += 1

    return plt, zone


#This function takes in the reference values and the prediction values as lists and returns a list with each index corresponding to the total number
#of points within that zone (0=A, 1=B, 2=C, 3=D, 4=E) and the plot
def clarke_error_grid(ref_values, pred_values, title_string):

    #Checking to see if the lengths of the reference and prediction arrays are the same
    assert (len(ref_values) == len(pred_values)), "Unequal number of values (reference : {}) (prediction : {}).".format(len(ref_values), len(pred_values))

    #Checks to see if the values are within the normal physiological range, otherwise it gives a warning
    if(max(ref_values) > 400 or max(pred_values) > 400):
        print("Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal physiological range of glucose (<400 mg/dl).", max(pred_values))
    if(min(ref_values) < 0 or min(pred_values) < 0):
        print("Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.", min(pred_values))

    #Clear plot
    plt.clf()

    #Set up plot
    plt.scatter(ref_values, pred_values, marker='o', color='blue', s=3)
    plt.title(title_string + " Clarke Error Grid")
    plt.xlabel("Reference Concentration [mg/dL]")
    plt.ylabel("Prediction Concentration [mg/dL]")
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.gca().set_facecolor('white')

    #Set axes lengths
    plt.gca().set_xlim([0, 400])
    plt.gca().set_ylim([0, 400])
    plt.gca().set_aspect((400)/(400))

    #Plot zone lines
    plt.plot([0,400], [0,400], ':', c='black')                      #Theoretical 45 regression line
    plt.plot([0, 175/3], [70, 70], '-', c='black')
    #plt.plot([175/3, 320], [70, 400], '-', c='black')
    plt.plot([175/3, 400/1.2], [70, 400], '-', c='black')           #Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
    plt.plot([70, 70], [84, 400],'-', c='black')
    plt.plot([0, 70], [180, 180], '-', c='black')
    plt.plot([70, 290],[180, 400],'-', c='black')
    # plt.plot([70, 70], [0, 175/3], '-', c='black')
    plt.plot([70, 70], [0, 56], '-', c='black')                     #Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
    # plt.plot([70, 400],[175/3, 320],'-', c='black')
    plt.plot([70, 400], [56, 320],'-', c='black')
    plt.plot([180, 180], [0, 70], '-', c='black')
    plt.plot([180, 400], [70, 70], '-', c='black')
    plt.plot([240, 240], [70, 180],'-', c='black')
    plt.plot([240, 400], [180, 180], '-', c='black')
    plt.plot([130, 180], [0, 70], '-', c='black')

    #Add zone titles
    plt.text(30, 15, "A", fontsize=15)
    plt.text(370, 260, "B", fontsize=15)
    plt.text(280, 370, "B", fontsize=15)
    plt.text(160, 370, "C", fontsize=15)
    plt.text(160, 15, "C", fontsize=15)
    plt.text(30, 140, "D", fontsize=15)
    plt.text(370, 120, "D", fontsize=15)
    plt.text(30, 370, "E", fontsize=15)
    plt.text(370, 15, "E", fontsize=15)

    #Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        if (ref_values[i] <= 70 and pred_values[i] <= 70) or (pred_values[i] <= 1.2*ref_values[i] and pred_values[i] >= 0.8*ref_values[i]):
            zone[0] += 1    #Zone A

        elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (ref_values[i] <= 70 and pred_values[i] >= 180):
            zone[4] += 1    #Zone E

        elif ((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110) or ((ref_values[i] >= 130 and ref_values[i] <= 180) and (pred_values[i] <= (7/5)*ref_values[i] - 182)):
            zone[2] += 1    #Zone C
        elif (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180)) or (ref_values[i] <= 175/3 and pred_values[i] <= 180 and pred_values[i] >= 70) or ((ref_values[i] >= 175/3 and ref_values[i] <= 70) and pred_values[i] >= (6/5)*ref_values[i]):
            zone[3] += 1    #Zone D
        else:
            zone[1] += 1    #Zone B

    return plt, zone


