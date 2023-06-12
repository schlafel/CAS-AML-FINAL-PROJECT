# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../src'))

import recommonmark
from recommonmark.transform import AutoStructify

# -- Project information -----------------------------------------------------

project = 'American Sign Language Recognition'
copyright = '2023, Asad Bin Imtiaz, Felix Schlatter'
author = 'Asad Bin Imtiaz, Felix Schlatter'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
'sphinxcontrib.bibtex',
    #'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'recommonmark'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


autodoc_mock_imports = ['torch',
                        'tensorflow',
                        'mediapipe',
                        'torchvision']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for LaTeX output ---------------------------------------------
latex_engine = 'pdflatex'
latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
'papersize': 'a4paper',
'releasename':" ",
# Sonny, Lenny, Glenn, Conny, Rejne, Bjarne and Bjornstrup
# 'fncychap': '\\usepackage[Lenny]{fncychap}',
'fncychap': '\\usepackage{fncychap}',
'fontpkg': '\\usepackage{amsmath,amsfonts,amssymb,amsthm}',
    'printindex': r'\footnotesize\raggedright\printindex',
'figure_align':'H',
''
# The font size ('10pt', '11pt' or '12pt').
#
'pointsize': '10pt',
'classoptions':',twocolumn,',
    'extraclassoptions': 'openany,oneside',
# Additional stuff for the LaTeX preamble.
#
'preamble': r'''
%% %% %% %% %% %% %% %% %% %% CASAML %% %% %% %% %% %% %% %% %%
%% %add number to subsubsection 2=subsection, 3=subsubsection
%% % below subsubsection is not good idea.
\setcounter{secnumdepth}{3}
%
%% %% Table of content upto 2=subsection, 3=subsubsection
\setcounter{tocdepth}{2}
\usepackage{amsmath,amsfonts,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{here}
\usepackage{blindtext}
\usepackage{svg}
%% % reduce spaces for Table of contents, figures and tables
%% % i t is used "\addtocontents{toc}{\vskip -1.2cm}" etc. in the document
\usepackage[notlot,nottoc,notlof]{}
\usepackage{color}
\usepackage{transparent}
\usepackage{eso-pic}
\usepackage{lipsum}
\usepackage{footnotebackref} %% link at the footnote to go to the place of footnote in the text


%----------------------------------------------------------------------------------------
%	THESIS INFORMATION
%----------------------------------------------------------------------------------------

% Those information will be used to generate the title page and the abstract. You can also access this information with the corresponding keyword. 


% Keywords used in the template. 
\newcommand{\thesistitle}{American Sign Language Recognition with DeepLearning} % Your thesis title, print it with \ttitle
\newcommand{\supervisor}{ } % Your supervisor's name, print it with \supname. Include academic title.
\newcommand{\coadvisor}{ } %Your co-advisors's name, print it with \coaname. Include academic title.
\newcommand{\firstname}{John} % Your first name, print it with \fname. 
\newcommand{\lastname}{Smith} % Your last name, print it with \lname .
\newcommand{\authorname}{\fname~ \lname} % Your whole name made from your input, print it with \authorname.
\newcommand{\matriculationnumber}{19-271-631} % Your matriculation number, print it with \matrnumber
\newcommand{\degree}{Certificate of Advanced Studies in Advanced Machine Learning AML University of Bern (CAS AML Unibe)} % Your degree name, print it with \degreename. 
%Possible degrees at the GCB: PhD in Biochemistry and Molecular Biology, PhD in Cell Biology, PhD in Biomedical Engineering, PhD in Biomedical Sciences, PhD in Immunology, PhD in Neuroscience, Doctor of Medicine and Philosophy (MD,PhD), Doctor of Veterinary Medicine and Philosophy (DVM,PhD), Doctor of Dentistry and Philosophy (DDS,PhD), PhD in Computational Biology, 
\newcommand{\institute}{Mathematical Institute} % Your department's name and URL, print it with \instname
\newcommand{\faculty}{} % Your department's name and URL, print it with \facname. Possible faculties: Faculty of Medicine, Faculty of Science, Vetsuisse Faculty, Institute of Virology and Immunology IVI, Mittelhäusern, Institute for Research in Biomedicine IRB, Bellinzona
\newcommand{\university}{University of Bern} % Your university's name and URL, print it with \univname. Possible Universities: University of Bern or University of Zurich (for Vetsuisse Zürich) 

% The following keywords are not used in this template. But define it if you want to use it in the text. 
\newcommand{\department}{University of Bern}% Your department's name and URL, print it with \deptname
\newcommand{\group}{asdfasdf} % Your research group's name and URL, print it with \groupname
\newcommand{\subject}{Cell biology} % Your subject area, print it with \subjectname
\newcommand{\keywords}{} % Keywords for your thesis, print it with \keywordnames





%% spacing between line
\usepackage{setspace}
%% %% \onehalfspacing
%% %% \doublespacing
\singlespacing
%% %% %% %% %% % d atetime
\usepackage{datetime}
\newdateformat{MonthYearFormat}{%
\monthname[\THEMONTH], \THEYEAR}
%% RO, LE will not work for 'oneside' layout.
%% Change oneside to twoside in document class
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
%% % Alternating Header for oneside
\fancyhead[L]{\ifthenelse{\isodd{\value{page}}}{ \small \nouppercase{\leftmark} }{}}
\fancyhead[R]{\ifthenelse{\isodd{\value{page}}}{}{ \small \nouppercase{\rightmark} }}
%% % Alternating Header for two side
%\fancyhead[RO]{\small \nouppercase{\rightmark}}
%\fancyhead[LE]{\small \nouppercase{\leftmark}}
%% for oneside: change footer at right side. If you want to use Left and right then use same as␣header defined above.
\fancyfoot[R]{\ifthenelse{\isodd{\value{page}}}{{\tiny Meher Krishna Patel} }{\href{http://pythondsp.readthedocs.io/en/latest/pythondsp/toc.html}{\tiny PythonDSP}}}
%% % Alternating Footer for two side
%\fancyfoot[RO, RE]{\scriptsize Meher Krishna Patel (mekrip@gmail.com)}
%% % page number
\fancyfoot[CO, CE]{\thepage}
\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\footrulewidth}{0.5pt}
\RequirePackage{tocbibind} %% % c omment this to remove page number for following
\addto\captionsenglish{\renewcommand{\contentsname}{Table of contents}}
\addto\captionsenglish{\renewcommand{\listfigurename}{List of figures}}
\addto\captionsenglish{\renewcommand{\listtablename}{List of tables}}
% \addto\captionsenglish{\renewcommand{\chaptername}{Chapter}}
%% reduce spacing for itemize
\usepackage{enumitem}
\setlist{nosep}
%% %% %% %% %% % Quote Styles at the top of chapter
\usepackage{epigraph}
\setlength{\epigraphwidth}{0.8\columnwidth}
\newcommand{\chapterquote}[2]{\epigraphhead[60]{\epigraph{\textit{#1}}{\textbf {\textit{--#2}}}}}
%% %% %% %% %% % Quote for all places except Chapter
\newcommand{\sectionquote}[2]{{\quote{\textit{``#1''}}{\textbf {\textit{--#2}}}}}
\usepackage{chngcntr}

\counterwithin*{section}{chapter}  % Reset section counter at each chapter
\titleformat{\section}[hang]{\normalfont\Large\bfseries}{\thesection}{1em}{} 
\renewcommand{\thesection}{\arabic{section}} %include chapter number in section title


''',

'maketitle': r'''
\pagenumbering{Roman} %% % to avoid page 1 conflict with actual page 1
\begin{titlepage}
{
\vspace*{-3cm}
\hfill
\includegraphics[scale= 0.7]{Logo_UniBe.pdf} % University logo
}
 \begin{center}
\hypersetup{hidelinks} 
 \vspace*{.06\textheight}
  {\Large CAS Advanced Machine Learning 2022\par} % graduate school name
 {\LARGE \university\par}\vspace{1.5cm} % University name
% \textsc{\Final Project}\\[0.5cm] % Thesis type

 {\huge \bfseries \thesistitle\par}\vspace{0.4cm} % Thesis title
  {\Large CAS Final Project}\\[0.5cm]
  {\LARGE \bfseries Felix Schlatter \& Asad Bin Imtiaz \par}\vspace{0.4cm} % Author name 
  {\Large for the degree of}\\[0.5cm]
  {\Large \degree \par}

   \vfill

 % \emph{Supervisor}\\[1mm]
 %{\Large \supervisor}\\ % Supervisor name
 %{\Large \institute}\\ % by default it is your Institute, change if necessary
 %{\Large  \faculty ~of the \university} \\% by default it is your Faculty, change if necessary

\vspace{0.4cm}
 %\emph{Co-advisor}\\[1mm]
 %{\Large asdf}\\ % Supervisor name
 %{\Large \href{https://www.mendel.uni.com/}{Institute of Gregor Mendel}}\\
 %{\Large \href{https://www.sci.mendel.uni.com/}{Faculty of Science} of the \href{https://www.uni.com/}{University of Mendel}}
 
 \vfill
 \end{center}
\end{titlepage}


\clearpage
\pagenumbering{roman}
\tableofcontents
\listoffigures
\listoftables
\clearpage
\pagenumbering{arabic}
''',
# Latex figure align
# 'figure_align': 'htbp',
'sphinxsetup': \
'hmargin={0.7in,0.7in}, vmargin={1in,1in}, \
verbatimwithframe=true, \
TitleColor={rgb}{0,0,0}, \
HeaderFamily=\\rmfamily\\bfseries, \
InnerLinkColor={rgb}{0,0,1}, \
OuterLinkColor={rgb}{0,0,1}',
#'tableofcontents':' ',
    'tableofcontents': '\\tableofcontents\n\n',  # Exclude bibliography from TOC

}


latex_logo = '_static/Logo_UniBe.pdf'
# latex_logo = '_static/logo.png'


bibtex_bibfiles = ['references.bib']

# Other configuration settings...

latex_index_module = 'sphinxcontrib.peculiarity'

