********************************************************************************
*                                              					
*    Machine Learning Methods for Demand Estimation
*             
*    Patrick Bajari, Denis Nekipelov, Stephen Ryan and Miaoyu Yang
*              
*    Updated: 2015-1-20    
*                                              					
********************************************************************************

clear all
set more off

********************************************************************************
* 1.0 BUILD DATA FOR R MAIN MODELS
********************************************************************************
* build data from scratch
* do 1_create_data.do

* read data into memory if the data is not built from scratch
use salty_snack_with_price_whole,clear 

* generate log price and quantity
gen logunits = log(units)
for var price*: gen logX = log(X)
*drop price*

* replace pissing logprice with code -9999
for var logprice*: replace X = -9999 if mi(X)

* spilt the data set into training(j=1), validation (j=2) set and test(j=0) set 
gen j = runiform()
replace j = 1 if j >= 0.4
replace j = 2 if j < 0.4 & j >= 0.25
replace j = 0 if j < 0.25

* consolidate values in some variables
decode flavor,gen(flavor)
replace flavor = "BBQ" 		if strpos(flavor,"BBQ")
replace flavor = "BBQ" 		if strpos(flavor,"BARB")
replace flavor = "BBQ" 		if strpos(flavor,"BRB")
replace flavor = "CHEDDAR" 	if strpos(flavor,"CHED")
replace flavor = "CHEDDAR" 	if strpos(flavor,"CHDR")
replace flavor = "ONION" 	if strpos(flavor,"ONI")
replace flavor = "ONION" 	if strpos(flavor,"ONN")
replace flavor = "VINEGAR" 	if strpos(flavor,"VINEGAR")
replace flavor = "CHEESE" 	if strpos(flavor,"CHEESE")
replace flavor = "CHEESE" 	if strpos(flavor,"PARMESAN")
replace flavor = "CHEESE" 	if strpos(flavor,"MOZZ")
replace flavor = "CHEESE" 	if strpos(flavor,"CHEDDAR")
replace flavor = "PEPPER" 	if strpos(flavor,"PEPPER")
replace flavor = "PEPPER" if strpos(flavor,"PPR")
replace flavor = "PEPPER" 	if strpos(flavor,"CHIL")
replace flavor = "SPICY" 	if strpos(flavor,"JALAPEN")
replace flavor = "SPICY" 	if strpos(flavor,"HOT")
replace flavor = "SPICY" 	if strpos(flavor,"SPICY")
replace flavor = "SPICY" 	if strpos(flavor,"WASABI")
replace flavor = "SPICY" 	if strpos(flavor,"HABANERO")
replace flavor = "SPICY" 	if strpos(flavor,"PEPPER")
replace flavor = "PIZZA" 	if strpos(flavor,"PIZZA")
replace flavor = "RANCH" 	if strpos(flavor,"RANCH")
replace flavor = "VARIETY" 	if strpos(flavor,"VARIETY")
replace flavor = "VARIETY" 	if strpos(flavor,"ASSORT")
replace flavor = "SMOKE" 	if strpos(flavor,"SMOK")
replace flavor = "SOUR CREAM" if strpos(flavor,"SOUR")
replace flavor = "SALSA" 	if strpos(flavor,"SALS")
replace flavor = "WING" 	if strpos(flavor,"WING")
replace flavor = "SPICE" 	if strpos(flavor,"MESQUITE")
replace flavor = "SPICE" 	if strpos(flavor,"MSQT")
replace flavor = "SPICE" 	if strpos(flavor,"CHIVE")
replace flavor = "SPICE" 	if strpos(flavor,"SPICE")
replace flavor = "SPICE" 	if strpos(flavor,"CAJUN")
replace flavor = "SPICE" 	if strpos(flavor,"DILL")
replace flavor = "HONEY" 	if strpos(flavor,"HONEY")
replace flavor = "RUSSET" 	if strpos(flavor,"RUSSET")
replace flavor = "TOMATO" 	if strpos(flavor,"TOMATO")
replace flavor = "TOMATO" 	if strpos(flavor,"KETCHUP")
replace flavor = "POTATO" 	if strpos(flavor,"POTAT")
replace flavor = "CAJUN" 	if strpos(flavor,"CAJUN")
replace flavor = "SEA" 		if strpos(flavor,"SEA")
replace flavor = "ORIGINAL" if inlist(flavor,"CLASSIC","ORIGINAL","REGULAR","PLAIN","NATURAL")
replace flavor = "ORIGINAL" if strpos(flavor,"CLASSIC")
replace flavor = "OTHER" 	if inlist(flavor,"AU GRATIN","DARK","GERMAN","GOURMET")
replace flavor = "OTHER" 	if strpos(flavor,"RIBS")
replace flavor = "OTHER" 	if strpos(flavor,"BURGER")
replace flavor = "OTHER" 	if strpos(flavor,"CONEY")
replace flavor = "OTHER" 	if strpos(flavor,"MILK")
replace flavor = "OTHER" 	if strpos(flavor,"PESTO")
replace flavor = "OTHER" 	if strpos(flavor,"LIM")
replace flavor = "OTHER" 	if strpos(flavor,"WORK")
replace flavor = "OTHER" 	if strpos(flavor,"GUACAMOLE")
replace flavor = "OTHER" 	if strpos(flavor,"HAWAIIAN")
replace flavor = "OTHER" 	if strpos(flavor,"SWEET")
drop flavorscent
rename flavor flavorscent

decode fat, gen(fat)
replace fat = "FAT FREE" 	if strpos(fat,"FREE")
replace fat = "LESS FAT" 	if strpos(fat,"LESS")
replace fat = "LESS FAT" 	if strpos(fat,"REDUCE")
replace fat = "LESS FAT" 	if strpos(fat,"LOW")
replace fat = "LESS FAT" 	if strpos(fat,"LIGHT")
replace fat = "LESS FAT" 	if strpos(fat,"1")
replace fat = "REGULAR" 	if strpos(fat,"MISSING")
drop fatcontent
rename fat fatcontent

decode salt, gen(salt)
replace salt = "NO SALT" 	if strpos(salt,"NO")
replace salt = "NO SALT" 	if strpos(salt,"FREE")
replace salt = "NO SALT" 	if strpos(salt,"UN")
replace salt = "REGULAR" 	if salt == "SALTED"
replace salt = "REGULAR" 	if strpos(salt,"REGULAR")
replace salt = "LOW SALT" 	if strpos(salt,"LOW")
replace salt = "LOW SALT" 	if strpos(salt,"LIGHT")
replace salt = "LOW SALT" 	if strpos(salt,"REDUCE")
replace salt = "REGULAR" 	if strpos(salt,"SODIUM")
drop saltsodium
rename salt saltsodiumcontent

decode typeofcut, gen(cut)
rename typeofcut cut 

replace cut = "RIPPLE" 		if strpos(cut,"RIP")
replace cut = "RIPPLE" 		if strpos(cut,"RINKLE")
replace cut = "RIPPLE" 		if strpos(cut,"RIDGE")
replace cut = "RIPPLE" 		if strpos(cut,"DIPPETTES")
replace cut = "RIPPLE" 		if strpos(cut,"WAVY")
replace cut = "RIPPLE" 		if strpos(cut,"CURL")
replace cut = "RIPPLE" 		if strpos(cut,"GROOVY")
replace cut = "RIPPLE" 		if strpos(cut,"MARCELLE")
replace cut = "THICK" 		if strpos(cut,"THICK")
replace cut = "WAFFLE" 		if strpos(cut,"WAFFLE")
replace cut = "THIN" 		if strpos(cut,"THIN")
replace cut = "DIP" 		if strpos(cut,"DIP")
replace cut = "FLAT" 		if strpos(cut,"FLAT")
drop typeofcut
rename cut typeofcut

decode package, gen(pak)
rename package pak 
replace pak = "INDIVIDUAL" if strpos(pak,"INDIV")
replace pak = "BOX" if strpos(pak,"BOX")
replace pak = "BAG" if strpos(pak,"BAG")
replace pak = "CANISTER" if strpos(pak,"CANISTER")
replace pak = "CANISTER" if strpos(pak,"CNSTR")
replace pak = "OTHER" if strpos(pak,"TIN")
replace pak = "OTHER" if strpos(pak,"TUB")
replace pak = "OTHER" if strpos(pak,"TRAY")
replace pak = "OTHER" if strpos(pak,"BUCKET")
replace pak = "OTHER" if strpos(pak,"JAR")
replace pak = "OTHER" if strpos(pak,"VARIETY")
replace pak = "OTHER" if strpos(pak,"CONTAINER")
drop package
rename pak package

decode cooking, gen(cook)
rename cooking cook
replace cook = "FRY" if strpos(cook,"FRI")
replace cook = "FRY" if strpos(cook,"FRY")
replace cook = "KETTLE" if strpos(cook,"KETTL")
replace cook = "KETTLE" if strpos(cook,"KTTL")
replace cook = "KETTLE" if strpos(cook,"KTL")
replace cook = "HAND" if strpos(cook,"HAND")
replace cook = "HOME" if strpos(cook,"HOME")
replace cook = "HOME" if strpos(cook,"OLD")
replace cook = "HOME" if strpos(cook,"KITCHEN")
replace cook = "CRISPY" if strpos(cook,"CRISPY")
replace cook = "CRISPY" if strpos(cook,"CRUNCHY")
replace cook = "OTHER" if strpos(cook,"STYL")
replace cook = "OTHER" if strpos(cook,"OLD")
replace cook = "OTHER" if strpos(cook,"AMERICAN")
replace cook = "OTHER" if strpos(cook,"ASSORTED")
replace cook = "OTHER" if strpos(cook,"GLD")
replace cook = "OTHER" if strpos(cook,"DARK")
replace cook = "OTHER" if strpos(cook,"COOKED")
drop cooking
rename cook cookingmethod

* clean up
capture drop _merge

* save a dta file in Stata 12 format
*saveold salty_snack_main.dta,replace
save salty_snack_main.dta,replace
outsheet using "salty_snack_main.csv", comma replace


* 5% subset of the data frame to test the code
set seed 1
gen sub = runiform()
keep if sub <= 0.05
drop sub
* save a temp dta file in Stata 12 format (import in R) 
*saveold salty_snack_temp_0.05.dta,replace
save salty_snack_temp_0.05.dta,replace
outsheet using "salty_snack_temp_0.05.csv", comma replace


* 5%*5% subset
set seed 1
gen sub = runiform()
keep if sub <= 0.05
drop sub
* save a temp dta file in Stata 12 format (import in R) 
saveold salty_snack_temp_0.05_0.05.dta,replace

