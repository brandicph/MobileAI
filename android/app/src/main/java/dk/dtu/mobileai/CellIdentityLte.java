
package dk.dtu.mobileai;

import com.google.gson.annotations.Expose;
import com.google.gson.annotations.SerializedName;

public class CellIdentityLte {

    @SerializedName("mcc")
    @Expose
    private long mcc;
    @SerializedName("tac")
    @Expose
    private long tac;
    @SerializedName("earfcn")
    @Expose
    private long earfcn;
    @SerializedName("ci")
    @Expose
    private long ci;
    @SerializedName("mnc")
    @Expose
    private long mnc;
    @SerializedName("pci")
    @Expose
    private long pci;
    @SerializedName("cell_info_lte")
    @Expose
    private String cellInfoLte;

    public long getMcc() {
        return mcc;
    }

    public void setMcc(long mcc) {
        this.mcc = mcc;
    }

    public long getTac() {
        return tac;
    }

    public void setTac(long tac) {
        this.tac = tac;
    }

    public long getEarfcn() {
        return earfcn;
    }

    public void setEarfcn(long earfcn) {
        this.earfcn = earfcn;
    }

    public long getCi() {
        return ci;
    }

    public void setCi(long ci) {
        this.ci = ci;
    }

    public long getMnc() {
        return mnc;
    }

    public void setMnc(long mnc) {
        this.mnc = mnc;
    }

    public long getPci() {
        return pci;
    }

    public void setPci(long pci) {
        this.pci = pci;
    }

    public String getCellInfoLte() {
        return cellInfoLte;
    }

    public void setCellInfoLte(String cellInfoLte) {
        this.cellInfoLte = cellInfoLte;
    }
}