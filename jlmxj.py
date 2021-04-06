from IPython.display import display, HTML
with open('/home-2/jwei9@jhu.edu/mxj_jupyter.css') as f:
    css = f.read().replace(';', ' !important;')
display(HTML('<style type="text/css">%s</style>Customized changes loaded.'%css))
