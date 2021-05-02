function setup_mplhep()
    @eval begin
        using Conda
        Conda.add_channel("conda-forge")
        Conda.add("mplhep")
    end
end

using PyCall, PyPlot

@pyimport mplhep as hep
hep.set_style(hep.style.CMS)
hep.set_style(Dict("font.sans-serif" => "DejaVu Sans"))
matplotlib.rc("text", usetex=true)
#matplotlib.rc("text", usetex=false)

function annotate_cms(ax)
    hep.cms.text("Private Work"; fontsize=20, ax)
    hep.cms.lumitext(L"41.5 $\mathrm{fb^{-1}}$"; fontsize=16, ax)
    ax.text(.05, .05, L"1 Lepton, 6 Jets, $\ge$3 Btags";
        transform=ax.transAxes, fontsize=16)
end
