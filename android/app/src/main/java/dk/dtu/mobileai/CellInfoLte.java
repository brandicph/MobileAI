package dk.dtu.mobileai;

import com.google.gson.annotations.Expose;
import com.google.gson.annotations.SerializedName;

import java.sql.Timestamp;
import java.util.Date;

public class CellInfoLte {


    @SerializedName("id")
    @Expose
    private long id;

    @SerializedName("url")
    @Expose
    private String url;

    @SerializedName("timestamp")
    @Expose
    private String timestamp;

    @SerializedName("registered")
    @Expose
    private Boolean registered;

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public String getTimestamp() {
        return timestamp;
    }

    public String getUrl() {
        return url;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public void setTimestamp(String timestamp) {
        this.timestamp = timestamp;
    }

    public Boolean getRegistered() {
        return registered;
    }

    public void setRegistered(Boolean registered) {
        this.registered = registered;
    }

}