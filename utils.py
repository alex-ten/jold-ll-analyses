import matplotlib.pyplot as plt

def add_legend(handles, colors, labels, bba, **kwargs):
    '''Automatically add legend by passing lists of handles, colors, and labels
    all of the same size

    Parameters
    ----------
    handles : list-like
        List of handles. Valid values are 
        'line', 'line+marker', 'line+error', 'patch', 'marker'
    colors : list-like
        List of colors
    labels : list-like
        List of labels
    bba : list-like
        This will be passed on to the `bbox_to_anchor` argument of 
        matplotlib's legend() function
    '''
    pass