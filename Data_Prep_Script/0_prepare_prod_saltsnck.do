clear all 

cd "E:\Data\IRI\salty_snack"

insheet using "E:\Data\IRI\salty_snack\prod_saltsnck_potato_chip.csv", clear

* import delimited using prod_saltsnck.xls, firstrow clear

gen double colupc =  item+100000*vend+10000000000*ge+100000000000*sy

drop item vend ge sy l1 l2 l3 l4 level upc stubspec1431rc00004

rename l9 name

rename l5 brand

save E:\Data\IRI\salty_snack\prod_saltsnck_potato_chip.dta,replace

outsheet using "E:\Data\IRI\salty_snack\prod_saltsnck_potato_chip.csv", comma replace
