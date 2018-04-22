package dk.dtu.mobileai.models;

import com.google.gson.annotations.Expose;
import com.google.gson.annotations.SerializedName;

public class Measurement {

    @SerializedName("url")
    @Expose
    private String url;
    @SerializedName("id")
    @Expose
    private Integer id;
    /**
     *
     * (Required)
     *
     */
    @SerializedName("antenna")
    @Expose
    private Integer antenna;
    /**
     *
     * (Required)
     *
     */
    @SerializedName("signal")
    @Expose
    private Integer signal;
    /**
     *
     * (Required)
     *
     */
    @SerializedName("level")
    @Expose
    private Integer level;
    /**
     *
     * (Required)
     *
     */
    @SerializedName("asu")
    @Expose
    private Integer asu;
    /**
     *
     * (Required)
     *
     */
    @SerializedName("rsrp")
    @Expose
    private Integer rsrp;
    /**
     *
     * (Required)
     *
     */
    @SerializedName("rsrq")
    @Expose
    private Integer rsrq;
    /**
     *
     * (Required)
     *
     */
    @SerializedName("rssnr")
    @Expose
    private Integer rssnr;
    /**
     *
     * (Required)
     *
     */
    @SerializedName("cqi")
    @Expose
    private Integer cqi;
    /**
     *
     * (Required)
     *
     */
    @SerializedName("cell_id")
    @Expose
    private Integer cellId;
    /**
     *
     * (Required)
     *
     */
    @SerializedName("mcc")
    @Expose
    private Integer mcc;
    /**
     *
     * (Required)
     *
     */
    @SerializedName("mnc")
    @Expose
    private Integer mnc;
    /**
     *
     * (Required)
     *
     */
    @SerializedName("pci")
    @Expose
    private Integer pci;
    /**
     *
     * (Required)
     *
     */
    @SerializedName("tac")
    @Expose
    private Integer tac;
    @SerializedName("created_at")
    @Expose
    private Object createdAt;
    @SerializedName("updated_at")
    @Expose
    private Object updatedAt;
    /**
     *
     * (Required)
     *
     */
    @SerializedName("entity")
    @Expose
    private String entity;

    public String getUrl() {
        return url;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    /**
     *
     * (Required)
     *
     */
    public Integer getAntenna() {
        return antenna;
    }

    /**
     *
     * (Required)
     *
     */
    public void setAntenna(Integer antenna) {
        this.antenna = antenna;
    }

    /**
     *
     * (Required)
     *
     */
    public Integer getSignal() {
        return signal;
    }

    /**
     *
     * (Required)
     *
     */
    public void setSignal(Integer signal) {
        this.signal = signal;
    }

    /**
     *
     * (Required)
     *
     */
    public Integer getLevel() {
        return level;
    }

    /**
     *
     * (Required)
     *
     */
    public void setLevel(Integer level) {
        this.level = level;
    }

    /**
     *
     * (Required)
     *
     */
    public Integer getAsu() {
        return asu;
    }

    /**
     *
     * (Required)
     *
     */
    public void setAsu(Integer asu) {
        this.asu = asu;
    }

    /**
     *
     * (Required)
     *
     */
    public Integer getRsrp() {
        return rsrp;
    }

    /**
     *
     * (Required)
     *
     */
    public void setRsrp(Integer rsrp) {
        this.rsrp = rsrp;
    }

    /**
     *
     * (Required)
     *
     */
    public Integer getRsrq() {
        return rsrq;
    }

    /**
     *
     * (Required)
     *
     */
    public void setRsrq(Integer rsrq) {
        this.rsrq = rsrq;
    }

    /**
     *
     * (Required)
     *
     */
    public Integer getRssnr() {
        return rssnr;
    }

    /**
     *
     * (Required)
     *
     */
    public void setRssnr(Integer rssnr) {
        this.rssnr = rssnr;
    }

    /**
     *
     * (Required)
     *
     */
    public Integer getCqi() {
        return cqi;
    }

    /**
     *
     * (Required)
     *
     */
    public void setCqi(Integer cqi) {
        this.cqi = cqi;
    }

    /**
     *
     * (Required)
     *
     */
    public Integer getCellId() {
        return cellId;
    }

    /**
     *
     * (Required)
     *
     */
    public void setCellId(Integer cellId) {
        this.cellId = cellId;
    }

    /**
     *
     * (Required)
     *
     */
    public Integer getMcc() {
        return mcc;
    }

    /**
     *
     * (Required)
     *
     */
    public void setMcc(Integer mcc) {
        this.mcc = mcc;
    }

    /**
     *
     * (Required)
     *
     */
    public Integer getMnc() {
        return mnc;
    }

    /**
     *
     * (Required)
     *
     */
    public void setMnc(Integer mnc) {
        this.mnc = mnc;
    }

    /**
     *
     * (Required)
     *
     */
    public Integer getPci() {
        return pci;
    }

    /**
     *
     * (Required)
     *
     */
    public void setPci(Integer pci) {
        this.pci = pci;
    }

    /**
     *
     * (Required)
     *
     */
    public Integer getTac() {
        return tac;
    }

    /**
     *
     * (Required)
     *
     */
    public void setTac(Integer tac) {
        this.tac = tac;
    }

    public Object getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Object createdAt) {
        this.createdAt = createdAt;
    }

    public Object getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(Object updatedAt) {
        this.updatedAt = updatedAt;
    }

    /**
     *
     * (Required)
     *
     */
    public String getEntity() {
        return entity;
    }

    /**
     *
     * (Required)
     *
     */
    public void setEntity(String entity) {
        this.entity = entity;
    }

}

