Each blockette will have a file containing one or more examples for a SEED
string and an XML-SEED String in one or more versions.

Name each file
blockette###.txt where ### is the three digit number of the blockette.

The test script will parse the file and test the conversion to and from all
given representations of each example.

VALID PYTHON CODE NEEDED!

Naming convention:
- The first two letters are the number and a dash.
- Then SEED for the SEED string, XSEED for an XML-string for all versions.
- If there are more than one version for a blockette, include the version
  in the string after a dash.
- If there is more than one XSEED string all strings need to have a version.
- A new line after SEED/XSEED.
- SEED can only have one line following.
- A new line after one part of the example ends.
- No empty lines in the XML.
  
Examples:

# First example with just one example to test:
1-SEED
010009502.1121992,001,00:00:00.0000~1992,002,00:00:00.0000~1993,029~IRIS DMC~Data for 1992,001~

1-XSEED
<XML pretty string>'

# Second example with two examples to test and also two versions.:
1-SEED
010009502.1121992,001,00:00:00.0000~1992,002,00:00:00.0000~1993,029~IRIS DMC~Data for 1992,001~

1-XSEED-1.0
<XML pretty string>

1-XSEED-1.1
<XML pretty string>

2-SEED
010009502.1121992,001,00:00:00.0000~1992,002,00:00:00.0000~1993,029~IRIS DMC~Data for 1992,001~

2-XSEED-1.0
<XML pretty string>

2-XSEED-1.1
<XML pretty string>