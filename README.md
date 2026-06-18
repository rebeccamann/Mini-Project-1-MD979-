## Optimising Non-Pharmaceutical Intervention Policies for Respiratory Virus Outbreaks in the Presence of Future Vaccine Uncertainty

Python code for an age-structured comparmental SEIHRVD model for a respiratory virus outbreak in the UK. Vaccine uncertainty and deployment day can be varied to observe the effect on the outbreak. This can be found in `mySEIHRVDmodel.ipynb`.

Non-pharmaceutical interventions (NPIs) are then considered in the form of a lockdown, where the start, duration, and intensity of the lockdown can be varied.
A second variant can also be added after a given period of time, where those vaccinated against variant one may still be susceptible to variant two.

Finally, the cost of varying NPIs can be calculated - code in `costCalculations.ipynb` - where QALY loss and GDP loss are combined to give a cost function, from which optimal strategies can be determined. It is also possible to create triangle plots with this information, varying the policymaker's 'willingness to pay' and evaluating the best strategy given confidence levels in differing vaccine efficacies (represented by weighting terms, $w_1, w_2, w_3$).
