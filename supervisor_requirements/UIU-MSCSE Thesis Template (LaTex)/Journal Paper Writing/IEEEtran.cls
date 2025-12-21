%%
%% Simplified IEEEtran-compatible class for journal papers
%% Based on IEEE Transactions format
%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{IEEEtran}[2015/08/26 V1.8b IEEE Transactions Style]

% Process class options
\DeclareOption{journal}{\def\@IEEEmode{journal}}
\DeclareOption{conference}{\def\@IEEEmode{conference}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ExecuteOptions{journal}
\ProcessOptions\relax

\LoadClass[10pt,twocolumn]{article}

% Required packages
\RequirePackage{amsmath,amssymb,amsfonts}
\RequirePackage{algorithmic}
\RequirePackage{algorithm}
\RequirePackage{graphicx}
\RequirePackage{textcomp}
\RequirePackage{xcolor}
\RequirePackage{cite}
\RequirePackage{multirow}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{caption}
\RequirePackage{float}
\RequirePackage{url}
\RequirePackage{hyperref}

% IEEE page layout
\setlength{\paperwidth}{8.5in}
\setlength{\paperheight}{11in}
\setlength{\textwidth}{7in}
\setlength{\textheight}{9.25in}
\setlength{\topmargin}{-0.5in}
\setlength{\headheight}{0in}
\setlength{\headsep}{0.25in}
\setlength{\oddsidemargin}{-0.25in}
\setlength{\evensidemargin}{-0.25in}
\setlength{\columnsep}{0.25in}
\setlength{\parindent}{1em}
\setlength{\parskip}{0pt}

% IEEE author block commands
\newcommand{\IEEEauthorblockN}[1]{{\large #1}\\[0.5em]}
\newcommand{\IEEEauthorblockA}[1]{{\small\itshape #1}}
\newcommand{\IEEEPARstart}[2]{\noindent{\LARGE\bfseries #1}{\large #2}}

% Title formatting
\def\@maketitle{%
  \newpage
  \null
  \vskip 2em%
  \begin{center}%
    \let \footnote \thanks
    {\LARGE\bfseries \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
  \end{center}%
  \par
  \vskip 1.5em}

% IEEE Section formatting (Roman numerals)
\renewcommand\thesection{\Roman{section}}
\renewcommand\thesubsection{\Alph{subsection}}
\renewcommand\thesubsubsection{\arabic{subsubsection}}

\renewcommand\section{\@startsection {section}{1}{\z@}%
  {-2.5ex \@plus -1ex \@minus -.2ex}%
  {1.5ex \@plus .2ex}%
  {\normalfont\large\bfseries\centering}}

\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
  {-2ex \@plus -1ex \@minus -.2ex}%
  {1ex \@plus .2ex}%
  {\normalfont\normalsize\bfseries\itshape}}

\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
  {-1.5ex \@plus -1ex \@minus -.2ex}%
  {0.5ex \@plus .2ex}%
  {\normalfont\normalsize\itshape}}

% Abstract environment
\renewenvironment{abstract}{%
  \small
  \begin{center}%
    {\bfseries Abstract\vspace{-.5em}\vspace{\z@}}%
  \end{center}%
  \quotation
}{\endquotation}

% Keywords environment
\newenvironment{IEEEkeywords}{%
  \vspace{0.5em}
  \noindent\textbf{\textit{Index Terms---}}%
}{\par\vspace{1em}}

% IEEE-style captions
\captionsetup{font=small,labelfont=bf,labelsep=period}

% Table and figure numbering
\renewcommand{\thetable}{\Roman{table}}
\renewcommand{\thefigure}{\arabic{figure}}

\endinput
