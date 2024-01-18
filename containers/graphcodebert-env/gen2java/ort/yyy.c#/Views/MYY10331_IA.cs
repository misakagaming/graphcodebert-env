// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: MYY10331_IA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:41:42
//
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// using Statements
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
using System;
using com.ca.gen.vwrt;
using com.ca.gen.vwrt.types;
using com.ca.gen.vwrt.vdf;
using com.ca.gen.csu.exception;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// START OF IMPORT VIEW
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
namespace GEN.ORT.YYY
{
  /// <summary>
  /// Internal data view storage for: MYY10331_IA
  /// </summary>
  [Serializable]
  public class MYY10331_IA : ViewBase, IImportView
  {
    private static MYY10331_IA[] freeArray = new MYY10331_IA[30];
    private static int countFree = 0;
    
    // Entity View: IMP_REFERENCE
    //        Type: IYY1_SERVER_DATA
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpReferenceIyy1ServerDataUserid
    /// </summary>
    private char _ImpReferenceIyy1ServerDataUserid_AS;
    /// <summary>
    /// Attribute missing flag for: ImpReferenceIyy1ServerDataUserid
    /// </summary>
    public char ImpReferenceIyy1ServerDataUserid_AS {
      get {
        return(_ImpReferenceIyy1ServerDataUserid_AS);
      }
      set {
        _ImpReferenceIyy1ServerDataUserid_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpReferenceIyy1ServerDataUserid
    /// Domain: Text
    /// Length: 8
    /// Varying Length: N
    /// </summary>
    private string _ImpReferenceIyy1ServerDataUserid;
    /// <summary>
    /// Attribute for: ImpReferenceIyy1ServerDataUserid
    /// </summary>
    public string ImpReferenceIyy1ServerDataUserid {
      get {
        return(_ImpReferenceIyy1ServerDataUserid);
      }
      set {
        _ImpReferenceIyy1ServerDataUserid = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpReferenceIyy1ServerDataReferenceId
    /// </summary>
    private char _ImpReferenceIyy1ServerDataReferenceId_AS;
    /// <summary>
    /// Attribute missing flag for: ImpReferenceIyy1ServerDataReferenceId
    /// </summary>
    public char ImpReferenceIyy1ServerDataReferenceId_AS {
      get {
        return(_ImpReferenceIyy1ServerDataReferenceId_AS);
      }
      set {
        _ImpReferenceIyy1ServerDataReferenceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpReferenceIyy1ServerDataReferenceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ImpReferenceIyy1ServerDataReferenceId;
    /// <summary>
    /// Attribute for: ImpReferenceIyy1ServerDataReferenceId
    /// </summary>
    public string ImpReferenceIyy1ServerDataReferenceId {
      get {
        return(_ImpReferenceIyy1ServerDataReferenceId);
      }
      set {
        _ImpReferenceIyy1ServerDataReferenceId = value;
      }
    }
    // Entity View: IMP
    //        Type: IYY1_TYPE
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1TypeTinstanceId
    /// </summary>
    private char _ImpIyy1TypeTinstanceId_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1TypeTinstanceId
    /// </summary>
    public char ImpIyy1TypeTinstanceId_AS {
      get {
        return(_ImpIyy1TypeTinstanceId_AS);
      }
      set {
        _ImpIyy1TypeTinstanceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1TypeTinstanceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ImpIyy1TypeTinstanceId;
    /// <summary>
    /// Attribute for: ImpIyy1TypeTinstanceId
    /// </summary>
    public string ImpIyy1TypeTinstanceId {
      get {
        return(_ImpIyy1TypeTinstanceId);
      }
      set {
        _ImpIyy1TypeTinstanceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1TypeTreferenceId
    /// </summary>
    private char _ImpIyy1TypeTreferenceId_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1TypeTreferenceId
    /// </summary>
    public char ImpIyy1TypeTreferenceId_AS {
      get {
        return(_ImpIyy1TypeTreferenceId_AS);
      }
      set {
        _ImpIyy1TypeTreferenceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1TypeTreferenceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ImpIyy1TypeTreferenceId;
    /// <summary>
    /// Attribute for: ImpIyy1TypeTreferenceId
    /// </summary>
    public string ImpIyy1TypeTreferenceId {
      get {
        return(_ImpIyy1TypeTreferenceId);
      }
      set {
        _ImpIyy1TypeTreferenceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1TypeTkeyAttrText
    /// </summary>
    private char _ImpIyy1TypeTkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1TypeTkeyAttrText
    /// </summary>
    public char ImpIyy1TypeTkeyAttrText_AS {
      get {
        return(_ImpIyy1TypeTkeyAttrText_AS);
      }
      set {
        _ImpIyy1TypeTkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1TypeTkeyAttrText
    /// Domain: Text
    /// Length: 4
    /// Varying Length: N
    /// </summary>
    private string _ImpIyy1TypeTkeyAttrText;
    /// <summary>
    /// Attribute for: ImpIyy1TypeTkeyAttrText
    /// </summary>
    public string ImpIyy1TypeTkeyAttrText {
      get {
        return(_ImpIyy1TypeTkeyAttrText);
      }
      set {
        _ImpIyy1TypeTkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1TypeTsearchAttrText
    /// </summary>
    private char _ImpIyy1TypeTsearchAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1TypeTsearchAttrText
    /// </summary>
    public char ImpIyy1TypeTsearchAttrText_AS {
      get {
        return(_ImpIyy1TypeTsearchAttrText_AS);
      }
      set {
        _ImpIyy1TypeTsearchAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1TypeTsearchAttrText
    /// Domain: Text
    /// Length: 20
    /// Varying Length: N
    /// </summary>
    private string _ImpIyy1TypeTsearchAttrText;
    /// <summary>
    /// Attribute for: ImpIyy1TypeTsearchAttrText
    /// </summary>
    public string ImpIyy1TypeTsearchAttrText {
      get {
        return(_ImpIyy1TypeTsearchAttrText);
      }
      set {
        _ImpIyy1TypeTsearchAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1TypeTotherAttrText
    /// </summary>
    private char _ImpIyy1TypeTotherAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1TypeTotherAttrText
    /// </summary>
    public char ImpIyy1TypeTotherAttrText_AS {
      get {
        return(_ImpIyy1TypeTotherAttrText_AS);
      }
      set {
        _ImpIyy1TypeTotherAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1TypeTotherAttrText
    /// Domain: Text
    /// Length: 2
    /// Varying Length: N
    /// </summary>
    private string _ImpIyy1TypeTotherAttrText;
    /// <summary>
    /// Attribute for: ImpIyy1TypeTotherAttrText
    /// </summary>
    public string ImpIyy1TypeTotherAttrText {
      get {
        return(_ImpIyy1TypeTotherAttrText);
      }
      set {
        _ImpIyy1TypeTotherAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1TypeTotherAttrDate
    /// </summary>
    private char _ImpIyy1TypeTotherAttrDate_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1TypeTotherAttrDate
    /// </summary>
    public char ImpIyy1TypeTotherAttrDate_AS {
      get {
        return(_ImpIyy1TypeTotherAttrDate_AS);
      }
      set {
        _ImpIyy1TypeTotherAttrDate_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1TypeTotherAttrDate
    /// Domain: Date
    /// Length: 8
    /// </summary>
    private int _ImpIyy1TypeTotherAttrDate;
    /// <summary>
    /// Attribute for: ImpIyy1TypeTotherAttrDate
    /// </summary>
    public int ImpIyy1TypeTotherAttrDate {
      get {
        return(_ImpIyy1TypeTotherAttrDate);
      }
      set {
        _ImpIyy1TypeTotherAttrDate = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1TypeTotherAttrTime
    /// </summary>
    private char _ImpIyy1TypeTotherAttrTime_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1TypeTotherAttrTime
    /// </summary>
    public char ImpIyy1TypeTotherAttrTime_AS {
      get {
        return(_ImpIyy1TypeTotherAttrTime_AS);
      }
      set {
        _ImpIyy1TypeTotherAttrTime_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1TypeTotherAttrTime
    /// Domain: Time
    /// Length: 6
    /// </summary>
    private int _ImpIyy1TypeTotherAttrTime;
    /// <summary>
    /// Attribute for: ImpIyy1TypeTotherAttrTime
    /// </summary>
    public int ImpIyy1TypeTotherAttrTime {
      get {
        return(_ImpIyy1TypeTotherAttrTime);
      }
      set {
        _ImpIyy1TypeTotherAttrTime = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1TypeTotherAttrAmount
    /// </summary>
    private char _ImpIyy1TypeTotherAttrAmount_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1TypeTotherAttrAmount
    /// </summary>
    public char ImpIyy1TypeTotherAttrAmount_AS {
      get {
        return(_ImpIyy1TypeTotherAttrAmount_AS);
      }
      set {
        _ImpIyy1TypeTotherAttrAmount_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1TypeTotherAttrAmount
    /// Domain: Number
    /// Length: 17
    /// Decimal Places: 2
    /// Decimal Precision: Y
    /// </summary>
    private decimal _ImpIyy1TypeTotherAttrAmount;
    /// <summary>
    /// Attribute for: ImpIyy1TypeTotherAttrAmount
    /// </summary>
    public decimal ImpIyy1TypeTotherAttrAmount {
      get {
        return(_ImpIyy1TypeTotherAttrAmount);
      }
      set {
        _ImpIyy1TypeTotherAttrAmount = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public MYY10331_IA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public MYY10331_IA( MYY10331_IA orig )
    {
      ImpReferenceIyy1ServerDataUserid_AS = orig.ImpReferenceIyy1ServerDataUserid_AS;
      ImpReferenceIyy1ServerDataUserid = orig.ImpReferenceIyy1ServerDataUserid;
      ImpReferenceIyy1ServerDataReferenceId_AS = orig.ImpReferenceIyy1ServerDataReferenceId_AS;
      ImpReferenceIyy1ServerDataReferenceId = orig.ImpReferenceIyy1ServerDataReferenceId;
      ImpIyy1TypeTinstanceId_AS = orig.ImpIyy1TypeTinstanceId_AS;
      ImpIyy1TypeTinstanceId = orig.ImpIyy1TypeTinstanceId;
      ImpIyy1TypeTreferenceId_AS = orig.ImpIyy1TypeTreferenceId_AS;
      ImpIyy1TypeTreferenceId = orig.ImpIyy1TypeTreferenceId;
      ImpIyy1TypeTkeyAttrText_AS = orig.ImpIyy1TypeTkeyAttrText_AS;
      ImpIyy1TypeTkeyAttrText = orig.ImpIyy1TypeTkeyAttrText;
      ImpIyy1TypeTsearchAttrText_AS = orig.ImpIyy1TypeTsearchAttrText_AS;
      ImpIyy1TypeTsearchAttrText = orig.ImpIyy1TypeTsearchAttrText;
      ImpIyy1TypeTotherAttrText_AS = orig.ImpIyy1TypeTotherAttrText_AS;
      ImpIyy1TypeTotherAttrText = orig.ImpIyy1TypeTotherAttrText;
      ImpIyy1TypeTotherAttrDate_AS = orig.ImpIyy1TypeTotherAttrDate_AS;
      ImpIyy1TypeTotherAttrDate = orig.ImpIyy1TypeTotherAttrDate;
      ImpIyy1TypeTotherAttrTime_AS = orig.ImpIyy1TypeTotherAttrTime_AS;
      ImpIyy1TypeTotherAttrTime = orig.ImpIyy1TypeTotherAttrTime;
      ImpIyy1TypeTotherAttrAmount_AS = orig.ImpIyy1TypeTotherAttrAmount_AS;
      ImpIyy1TypeTotherAttrAmount = orig.ImpIyy1TypeTotherAttrAmount;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static MYY10331_IA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new MYY10331_IA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new MYY10331_IA());
          }
          else 
          {
            MYY10331_IA result = freeArray[--countFree];
            freeArray[countFree] = null;
            result.Reset(  );
            return(result);
          }
        }
      }
    }
    /// <summary>
    /// Static free instance method
    /// </summary>
    
    public void FreeInstance(  )
    {
      lock (freeArray)
      {
        if ( countFree < freeArray.Length )
        {
          freeArray[countFree++] = this;
        }
      }
    }
    /// <summary>
    /// clone constructor
    /// </summary>
    
    public override Object Clone(  )
    {
      return(new MYY10331_IA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      ImpReferenceIyy1ServerDataUserid_AS = ' ';
      ImpReferenceIyy1ServerDataUserid = "        ";
      ImpReferenceIyy1ServerDataReferenceId_AS = ' ';
      ImpReferenceIyy1ServerDataReferenceId = "00000000000000000000";
      ImpIyy1TypeTinstanceId_AS = ' ';
      ImpIyy1TypeTinstanceId = "00000000000000000000";
      ImpIyy1TypeTreferenceId_AS = ' ';
      ImpIyy1TypeTreferenceId = "00000000000000000000";
      ImpIyy1TypeTkeyAttrText_AS = ' ';
      ImpIyy1TypeTkeyAttrText = "    ";
      ImpIyy1TypeTsearchAttrText_AS = ' ';
      ImpIyy1TypeTsearchAttrText = "                    ";
      ImpIyy1TypeTotherAttrText_AS = ' ';
      ImpIyy1TypeTotherAttrText = "  ";
      ImpIyy1TypeTotherAttrDate_AS = ' ';
      ImpIyy1TypeTotherAttrDate = 00000000;
      ImpIyy1TypeTotherAttrTime_AS = ' ';
      ImpIyy1TypeTotherAttrTime = 00000000;
      ImpIyy1TypeTotherAttrAmount_AS = ' ';
      ImpIyy1TypeTotherAttrAmount = DecimalAttr.GetDefaultValue();
    }
    /// <summary>
    /// Gets the VDF version of the current state of the instance.
    /// </summary>
    public VDF GetVDF(  )
    {
      throw new Exception("can only execute GetVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current state of the instance to the VDF version.
    /// </summary>
    public void SetFromVDF( VDF vdf )
    {
      throw new Exception("can only execute SetFromVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( IImportView orig )
    {
      this.CopyFrom((MYY10331_IA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( MYY10331_IA orig )
    {
      ImpReferenceIyy1ServerDataUserid_AS = orig.ImpReferenceIyy1ServerDataUserid_AS;
      ImpReferenceIyy1ServerDataUserid = orig.ImpReferenceIyy1ServerDataUserid;
      ImpReferenceIyy1ServerDataReferenceId_AS = orig.ImpReferenceIyy1ServerDataReferenceId_AS;
      ImpReferenceIyy1ServerDataReferenceId = orig.ImpReferenceIyy1ServerDataReferenceId;
      ImpIyy1TypeTinstanceId_AS = orig.ImpIyy1TypeTinstanceId_AS;
      ImpIyy1TypeTinstanceId = orig.ImpIyy1TypeTinstanceId;
      ImpIyy1TypeTreferenceId_AS = orig.ImpIyy1TypeTreferenceId_AS;
      ImpIyy1TypeTreferenceId = orig.ImpIyy1TypeTreferenceId;
      ImpIyy1TypeTkeyAttrText_AS = orig.ImpIyy1TypeTkeyAttrText_AS;
      ImpIyy1TypeTkeyAttrText = orig.ImpIyy1TypeTkeyAttrText;
      ImpIyy1TypeTsearchAttrText_AS = orig.ImpIyy1TypeTsearchAttrText_AS;
      ImpIyy1TypeTsearchAttrText = orig.ImpIyy1TypeTsearchAttrText;
      ImpIyy1TypeTotherAttrText_AS = orig.ImpIyy1TypeTotherAttrText_AS;
      ImpIyy1TypeTotherAttrText = orig.ImpIyy1TypeTotherAttrText;
      ImpIyy1TypeTotherAttrDate_AS = orig.ImpIyy1TypeTotherAttrDate_AS;
      ImpIyy1TypeTotherAttrDate = orig.ImpIyy1TypeTotherAttrDate;
      ImpIyy1TypeTotherAttrTime_AS = orig.ImpIyy1TypeTotherAttrTime_AS;
      ImpIyy1TypeTotherAttrTime = orig.ImpIyy1TypeTotherAttrTime;
      ImpIyy1TypeTotherAttrAmount_AS = orig.ImpIyy1TypeTotherAttrAmount_AS;
      ImpIyy1TypeTotherAttrAmount = orig.ImpIyy1TypeTotherAttrAmount;
    }
  }
}
