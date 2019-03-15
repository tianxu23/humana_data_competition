setwd("C:/Users/31221/Desktop/Humana")
humana = read.csv("TAMU_FINAL_DATASET_2018.csv")
summary(humana)
summary(humana$Pct_above_poverty_line)
summary

humana[is.na(humana$Online_User)&(!(is.na('Education_level'))),]
test = humana[is.na(humana$Online_User),'Num_person_household']
head(test,500)
tail(test,500)
Pct_above_poverty_line
Pct_below_poverty_line
Home_value
Est_Net_worth
Est_income
Index_Health_ins_engage
Index_Health_ins_influence
Population_density_centile_ST
Population_density_centile_US
Dwelling_Type
Education_level
Length_residence
Est_BMI_decile
Num_person_household
College
