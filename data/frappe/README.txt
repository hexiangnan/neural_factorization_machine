Frappe dataset v1.0 http://baltrunas.info/research-menu/frappe

The frappe dataset contains a context-aware app usage log.
It consist of 96203 entries by 957 users for 4082 apps used in various contexts.
(sample 2 negative samples for 1 positive => # of total instances: 288609)

Nonzero u-i pairs: 18842
Context fields:
	#user:  957
	#item:  4082
	#cnt:  1981 (means how many times the app has been used by the user; convert it to 0/1)
	#daytime:  7
	#weekday:  7
	#isweekend:  2
	#homework:  3
	#cost:  2
	#weather:  9
	#country:  80
	#city:  233
Total features: 7363 - 1981 = 5382


Any scientific publications that use this data set should cite the following paper as the reference:
@Article{frappe15,
    title={Frappe: Understanding the Usage and Perception of Mobile App Recommendations In-The-Wild},
    author = {Linas Baltrunas, Karen Church, Alexandros Karatzoglou, Nuria Oliver},
    date={2015},
    urldate={2015-05-12},
    eprinttype={arxiv},
    eprint={arXiv:1505.03014}
}

Nobody guarantees the correctness of the data, its suitability for any particular purpose, 
or the validity of results based on the use of the data set. The data set may be used for any 
research purposes under the following conditions:
* The user must acknowledge the use of the data set in publications resulting from the use of the data set.
* The user may not redistribute the data without separate permission.
* The user may not try to deanonymise the data.
* The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from us.

In no event anyone involved in frappe project be liable to you for any damages arising out of the use or inability to use the 
associated scripts (including but not limited to loss of data or data being rendered inaccurate).

The three data files are encoded as UTF-8. You can use the following pandas script in python to load the data set:

import pandas
df = pandas.read_csv('frappe.csv', sep="\t")
meta_app = pandas.read_csv('apps.csv', sep="\t")
df = df.merge(meta_app, on='item')

Note, that we don't provide city names for privacy reasons. Also apps, that were downloaded not so many times are missing meta information.
However, the item id is a valid identification.

If you have any further questions or comments, please email linas.baltrunas@gmail.com