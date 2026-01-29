if __name__=="__main__":
    from juliacall import Main as jl
    jl.seval("import Pkg; Pkg.add(\"DataFrames\")")
    jl.seval("import Pkg; Pkg.add(\"CorrectMatch\")")
    import correctmatch