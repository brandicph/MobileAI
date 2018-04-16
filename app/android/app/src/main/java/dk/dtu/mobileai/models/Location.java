package dk.dtu.mobileai.models;

import com.google.gson.annotations.Expose;
import com.google.gson.annotations.SerializedName;

public class Location {

    @SerializedName("longitude")
    @Expose
    private double longitude;

    @SerializedName("latitude")
    @Expose
    private double latitude;

    @SerializedName("entity")
    @Expose
    private String entity;


    public double getLongitude() {
        return longitude;
    }

    public void setLongitude(double longitude) {
        this.longitude = longitude;
    }

    public double getLatitude() {
        return latitude;
    }


    public void setLatitude(double latitude) {
        this.latitude = latitude;
    }

    public String getEntity() {
        return entity;
    }


    public void setEntity(String entity) {
        this.entity = entity;
    }

}
