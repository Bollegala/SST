"""
Configurations of MLIB.
"""

# Separates comments in instance files.
COMMENT_DELIMITTER = '#'

# Separates attributes in instance files.
ATTRIBUTE_DELIMITTER = ' '

# Separates the attribute and its value.
VALUE_DELIMITTER = ':'

#Note: If the value delimitter also occurs in the feature id string,
# then we will only split at the last occurrence
# (just before the feature value) and consider the entire portion that
# precedes this point is the feature id.

# If this is set, we will not write zero features to files.
# This affects the feature files produced during scaling and held out selection.
DONT_WRITE_ZEROS = True 
