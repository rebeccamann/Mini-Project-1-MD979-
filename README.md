## Optimising Non-Pharmaceutical Intervention Policies for Respiratory Virus Outbreaks in the Presence of Future Vaccine Uncertainty

Python code for an age-structured comparmental SEIHRVD model for a respiratory virus outbreak in the UK. Vaccine uncertainty and deployment day can be varied to observe the effect on the outbreak. The parameters for the model are found in `modelParameters.ipynb`.

Non-pharmaceutical interventions (NPIs) are then considered in the form of a lockdown, where the start, duration, and intensity of the lockdown can be varied, as well as the possibility of two lockdowns. A second variant can also be added after a given period of time, where those vaccinated against variant one may still be susceptible to variant two. This model can be found in `SEIHRVDmodel.ipynb`.

Finally, the cost of varying NPIs can be calculated - code in `costCalculations.ipynb` - where QALY loss and GDP loss are combined to give a cost function, from which optimal strategies can be determined. This also considers an additional GDP loss when schools are closed during lockdown, to reflect the loss to the economy of parents not working to care for their children. From this, raw costs can be printed as well as graphs showing the net monetary loss against the duration of lockdown. This can be used to visualise which durations minimise the loss.

It is also possible to create triangle plots with this information, varying the policymaker's 'willingness to pay' and evaluating the best strategy given confidence levels in differing vaccine efficacies (represented by weighting terms, $w_1, w_2, w_3$). The code for this is found in `costCalculations.ipynb`, and the code to obtain the particular scenarios described in the full report are found in `trianglePlots.ipynb`.
