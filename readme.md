### Setting up the test suites for yourself
#### Prerequisites
A MySQL database locally hosted named "Matrices"
An auth.txt set up as the example auth_template.txt provided in this repository
Python3 with following modules installed:
- NumPy
- MySQL
- Matplotlib
#### Running the test suites
In testsuites.py provided in this repository for each TestsuiteX() there is a prerequisite commented for the table that needs to be created in the Matrices database
Afterwards the last line has to be adjusted to run TestsuiteX()
To generate the graphs the function TestsuiteXGraph() needs to be run after the tests have concluded
Note that some of the tests may run for more than a day depending on the machine they are running on
#### Patchnotes
Adjustments to the repository after finalizing the thesis will be collected here