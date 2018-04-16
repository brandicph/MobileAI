package dk.dtu.mobileai.models;

import com.google.gson.annotations.Expose;
import com.google.gson.annotations.SerializedName;

public class Entity {

    @SerializedName("url")
    @Expose
    private Object url;
    @SerializedName("locations")
    @Expose
    private Object locations;
    /**
     *
     * (Required)
     *
     */
    @SerializedName("name")
    @Expose
    private String name;
    /**
     *
     * (Required)
     *
     */
    @SerializedName("imei")
    @Expose
    private String imei;
    @SerializedName("created_at")
    @Expose
    private Object createdAt;
    @SerializedName("updated_at")
    @Expose
    private Object updatedAt;

    public Object getUrl() {
        return url;
    }

    public void setUrl(Object url) {
        this.url = url;
    }

    public Object getLocations() {
        return locations;
    }

    public void setLocations(Object locations) {
        this.locations = locations;
    }

    /**
     *
     * (Required)
     *
     */
    public String getName() {
        return name;
    }

    /**
     *
     * (Required)
     *
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     *
     * (Required)
     *
     */
    public String getImei() {
        return imei;
    }

    /**
     *
     * (Required)
     *
     */
    public void setImei(String imei) {
        this.imei = imei;
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

}
