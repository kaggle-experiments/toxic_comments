"""
    Local Utilities, Helper Functions

"""
from pprint import pprint, pformat


def squeeze(lol):
    """
    List of lists to List

    Args:
        lol : List of lists

    Returns:
       List 

    """
    return [ i for l in lol for i in l ]

"""
    util functions to enable pretty print on namedtuple

"""
def _namedtuple_repr_(self):
    return pformat(self.___asdict())

def ___asdict(self):
    d = self._asdict()
    for k, v in d.items():
        if hasattr(v, '_asdict'):
            d[k] = ___asdict(v)

    return dict(d)



class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of 
    the form [[1,2,3],[4,5,6]], and renders an HTML Table in 
    IPython Notebook. 
    Taken from http://calebmadrigal.com/display-list-as-table-in-ipython-notebook/"""
    
    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")
            
            for col in row:
                html.append("<td>{0}</td>".format(col))
            
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)