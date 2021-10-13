********************************************************************************
*                                              					
*    Machine Learning Methods for Demand Estimation
*             
*    Patrick Bajari, Denis Nekipelov, Stephen Ryan and Miaoyu Yang
*              
*    Updated: 2015-1-20    
*                                              					
********************************************************************************

********************************************************************************
*1.0 Create Store Aggregate Data
********************************************************************************
clear all
set more off

cd D:\Data\IRI\salty_snack

*read raw data and convert to Stata format
forvalues x = 1/6 {  
	insheet using saltsnck_groc_year`x'.raw,clear

	*generate upc code from item, vend, ge and sy
	gen double colupc=  item+100000*vend+10000000000*ge+100000000000*sy

	*bysort colupc: gen prod_share = _N


	*keep only potato chips using file prod_saltsnck.dta
	merge m:1 colupc using prod_saltsnck.dta , keepusing(colupc) keep(match) nogenerate

	gen year = `x'

	*save a temp file for each year
	save tempfile_year`x'.dta,replace
}

*combine 6 years
forvalues x = 1/6 {
	append using tempfile_year`x'
}

*keep necessary variables
keep colupc iri_key week dollars units pr f d

*generagte price variable
gen price = dollars/units

*merge in product names
merge m:1 colupc using prod_saltsnck_short,keep(match master) nogen
capture rename l9 name

*re-order variables
order iri_key week colupc name price units pr f d

*map IRI week to norminal week
gen nweek = week - 1113
drop week
rename nweek week

*map IRI week to actual date, year, month of year and week of month
gen date = (week+4157+1113)*7-5-21916
format date % td
gen year = year(date)
gen month = month(date)
gen numweek = week(date)

*only keep part of the data, drop tail observations that only has too few observations
bysort iri_key: gen iri_count = _N

*just enough to keep iri_key == 652159
drop if iri_count < 30554  

bysort iri_key colupc: gen prod_count = _N
drop if prod_count <= 52

bysort colupc :gen prodcount =  _N
drop if prodcount < 10000

drop iri_count prod_count prodcount

*generate short variable p_id for upc
egen p_id = group(colupc)

save chips_store_aggregate.dta,replace


********************************************************************************
* 2.0 Include Competitor Prices
********************************************************************************
*use cd D:\Data\IRI\salty_snack\chips_store_aggregate.dta,clear

*Change to working directory
cd /this_folder/

*Keep approx. 25% of the stores
sum iri_key,d
keep if iri_key<=r(p25)

*Drop variables to save space
drop name p_id

*Merge in product information
merge m:1 colupc using prod_saltsnck_category,keep(match) nogen

egen yrmonth = group(year month)
*save salty_snack_whole.dta,replace

*generate price for monthly weighted (by quantity) average price for each product
preserve
egen yrmonth = group(year month)
bysort colupc yrmonth: egen tot_units= total(units)
bysort colupc yrmonth: egen tot_revenue = total(price*units)
bysort colupc yrmonth: gen mean_price = tot_revenue/tot_units
keep colupc yrmonth mean_price
duplicates drop colupc yrmonth,force
rename mean_price price
reshape wide price, i(yrmonth) j(colupc)
save data/salty_snack_price.dta,replace
restore

*merge price info into main data
merge m:1 yrmonth using data/salty_snack_price.dta,nogen keep(master match)

*replace missing prices with 0
for var price*: replace X=0 if X ==.

*save a version of data with competitor prices
save data/salty_snack_with_price_whole.dta,replace

