package dk.dtu.mobileai;

import com.google.gson.annotations.Expose;
import com.google.gson.annotations.SerializedName;

public class CellSignalStrengthLte {

    @SerializedName("cqi")
    @Expose
    private long cqi;
    @SerializedName("rssnr")
    @Expose
    private long rssnr;
    @SerializedName("ta")
    @Expose
    private long ta;
    @SerializedName("ss")
    @Expose
    private long ss;
    @SerializedName("rsrp")
    @Expose
    private long rsrp;
    @SerializedName("rsrq")
    @Expose
    private long rsrq;
    @SerializedName("cell_info_lte")
    @Expose
    private String cellInfoLte;

    public long getCqi() {
        return cqi;
    }

    public void setCqi(long cqi) {
        this.cqi = cqi;
    }

    public long getRssnr() {
        return rssnr;
    }

    public void setRssnr(long rssnr) {
        this.rssnr = rssnr;
    }

    public long getTa() {
        return ta;
    }

    public void setTa(long ta) {
        this.ta = ta;
    }

    public long getSs() {
        return ss;
    }

    public void setSs(long ss) {
        this.ss = ss;
    }

    public long getRsrp() {
        return rsrp;
    }

    public void setRsrp(long rsrp) {
        this.rsrp = rsrp;
    }

    public long getRsrq() {
        return rsrq;
    }

    public void setRsrq(long rsrq) {
        this.rsrq = rsrq;
    }

    public String getCellInfoLte() {
        return cellInfoLte;
    }

    public void setCellInfoLte(String cellInfoLte) {
        this.cellInfoLte = cellInfoLte;
    }

}