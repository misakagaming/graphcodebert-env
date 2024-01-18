// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: MYY10121_LA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:42:01
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
// START OF LOCAL VIEW
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
namespace GEN.ORT.YYY
{
  /// <summary>
  /// Internal data view storage for: MYY10121_LA
  /// </summary>
  [Serializable]
  public class MYY10121_LA : ViewBase, ILocalView
  {
    private static MYY10121_LA[] freeArray = new MYY10121_LA[30];
    private static int countFree = 0;
    
    // Entity View: LOC_IMP
    //        Type: PARENT
    /// <summary>
    /// Internal storage for attribute missing flag for: LocImpParentPinstanceId
    /// </summary>
    private char _LocImpParentPinstanceId_AS;
    /// <summary>
    /// Attribute missing flag for: LocImpParentPinstanceId
    /// </summary>
    public char LocImpParentPinstanceId_AS {
      get {
        return(_LocImpParentPinstanceId_AS);
      }
      set {
        _LocImpParentPinstanceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocImpParentPinstanceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _LocImpParentPinstanceId;
    /// <summary>
    /// Attribute for: LocImpParentPinstanceId
    /// </summary>
    public string LocImpParentPinstanceId {
      get {
        return(_LocImpParentPinstanceId);
      }
      set {
        _LocImpParentPinstanceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocImpParentPkeyAttrText
    /// </summary>
    private char _LocImpParentPkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: LocImpParentPkeyAttrText
    /// </summary>
    public char LocImpParentPkeyAttrText_AS {
      get {
        return(_LocImpParentPkeyAttrText_AS);
      }
      set {
        _LocImpParentPkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocImpParentPkeyAttrText
    /// Domain: Text
    /// Length: 5
    /// Varying Length: N
    /// </summary>
    private string _LocImpParentPkeyAttrText;
    /// <summary>
    /// Attribute for: LocImpParentPkeyAttrText
    /// </summary>
    public string LocImpParentPkeyAttrText {
      get {
        return(_LocImpParentPkeyAttrText);
      }
      set {
        _LocImpParentPkeyAttrText = value;
      }
    }
    // Entity View: LOC_EXP
    //        Type: PARENT
    /// <summary>
    /// Internal storage for attribute missing flag for: LocExpParentPinstanceId
    /// </summary>
    private char _LocExpParentPinstanceId_AS;
    /// <summary>
    /// Attribute missing flag for: LocExpParentPinstanceId
    /// </summary>
    public char LocExpParentPinstanceId_AS {
      get {
        return(_LocExpParentPinstanceId_AS);
      }
      set {
        _LocExpParentPinstanceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocExpParentPinstanceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _LocExpParentPinstanceId;
    /// <summary>
    /// Attribute for: LocExpParentPinstanceId
    /// </summary>
    public string LocExpParentPinstanceId {
      get {
        return(_LocExpParentPinstanceId);
      }
      set {
        _LocExpParentPinstanceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocExpParentPreferenceId
    /// </summary>
    private char _LocExpParentPreferenceId_AS;
    /// <summary>
    /// Attribute missing flag for: LocExpParentPreferenceId
    /// </summary>
    public char LocExpParentPreferenceId_AS {
      get {
        return(_LocExpParentPreferenceId_AS);
      }
      set {
        _LocExpParentPreferenceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocExpParentPreferenceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _LocExpParentPreferenceId;
    /// <summary>
    /// Attribute for: LocExpParentPreferenceId
    /// </summary>
    public string LocExpParentPreferenceId {
      get {
        return(_LocExpParentPreferenceId);
      }
      set {
        _LocExpParentPreferenceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocExpParentPcreateUserId
    /// </summary>
    private char _LocExpParentPcreateUserId_AS;
    /// <summary>
    /// Attribute missing flag for: LocExpParentPcreateUserId
    /// </summary>
    public char LocExpParentPcreateUserId_AS {
      get {
        return(_LocExpParentPcreateUserId_AS);
      }
      set {
        _LocExpParentPcreateUserId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocExpParentPcreateUserId
    /// Domain: Text
    /// Length: 8
    /// Varying Length: N
    /// </summary>
    private string _LocExpParentPcreateUserId;
    /// <summary>
    /// Attribute for: LocExpParentPcreateUserId
    /// </summary>
    public string LocExpParentPcreateUserId {
      get {
        return(_LocExpParentPcreateUserId);
      }
      set {
        _LocExpParentPcreateUserId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocExpParentPupdateUserId
    /// </summary>
    private char _LocExpParentPupdateUserId_AS;
    /// <summary>
    /// Attribute missing flag for: LocExpParentPupdateUserId
    /// </summary>
    public char LocExpParentPupdateUserId_AS {
      get {
        return(_LocExpParentPupdateUserId_AS);
      }
      set {
        _LocExpParentPupdateUserId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocExpParentPupdateUserId
    /// Domain: Text
    /// Length: 8
    /// Varying Length: N
    /// </summary>
    private string _LocExpParentPupdateUserId;
    /// <summary>
    /// Attribute for: LocExpParentPupdateUserId
    /// </summary>
    public string LocExpParentPupdateUserId {
      get {
        return(_LocExpParentPupdateUserId);
      }
      set {
        _LocExpParentPupdateUserId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocExpParentPkeyAttrText
    /// </summary>
    private char _LocExpParentPkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: LocExpParentPkeyAttrText
    /// </summary>
    public char LocExpParentPkeyAttrText_AS {
      get {
        return(_LocExpParentPkeyAttrText_AS);
      }
      set {
        _LocExpParentPkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocExpParentPkeyAttrText
    /// Domain: Text
    /// Length: 5
    /// Varying Length: N
    /// </summary>
    private string _LocExpParentPkeyAttrText;
    /// <summary>
    /// Attribute for: LocExpParentPkeyAttrText
    /// </summary>
    public string LocExpParentPkeyAttrText {
      get {
        return(_LocExpParentPkeyAttrText);
      }
      set {
        _LocExpParentPkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocExpParentPsearchAttrText
    /// </summary>
    private char _LocExpParentPsearchAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: LocExpParentPsearchAttrText
    /// </summary>
    public char LocExpParentPsearchAttrText_AS {
      get {
        return(_LocExpParentPsearchAttrText_AS);
      }
      set {
        _LocExpParentPsearchAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocExpParentPsearchAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _LocExpParentPsearchAttrText;
    /// <summary>
    /// Attribute for: LocExpParentPsearchAttrText
    /// </summary>
    public string LocExpParentPsearchAttrText {
      get {
        return(_LocExpParentPsearchAttrText);
      }
      set {
        _LocExpParentPsearchAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocExpParentPotherAttrText
    /// </summary>
    private char _LocExpParentPotherAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: LocExpParentPotherAttrText
    /// </summary>
    public char LocExpParentPotherAttrText_AS {
      get {
        return(_LocExpParentPotherAttrText_AS);
      }
      set {
        _LocExpParentPotherAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocExpParentPotherAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _LocExpParentPotherAttrText;
    /// <summary>
    /// Attribute for: LocExpParentPotherAttrText
    /// </summary>
    public string LocExpParentPotherAttrText {
      get {
        return(_LocExpParentPotherAttrText);
      }
      set {
        _LocExpParentPotherAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocExpParentPtypeTkeyAttrText
    /// </summary>
    private char _LocExpParentPtypeTkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: LocExpParentPtypeTkeyAttrText
    /// </summary>
    public char LocExpParentPtypeTkeyAttrText_AS {
      get {
        return(_LocExpParentPtypeTkeyAttrText_AS);
      }
      set {
        _LocExpParentPtypeTkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocExpParentPtypeTkeyAttrText
    /// Domain: Text
    /// Length: 4
    /// Varying Length: N
    /// </summary>
    private string _LocExpParentPtypeTkeyAttrText;
    /// <summary>
    /// Attribute for: LocExpParentPtypeTkeyAttrText
    /// </summary>
    public string LocExpParentPtypeTkeyAttrText {
      get {
        return(_LocExpParentPtypeTkeyAttrText);
      }
      set {
        _LocExpParentPtypeTkeyAttrText = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public MYY10121_LA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public MYY10121_LA( MYY10121_LA orig )
    {
      LocImpParentPinstanceId_AS = orig.LocImpParentPinstanceId_AS;
      LocImpParentPinstanceId = orig.LocImpParentPinstanceId;
      LocImpParentPkeyAttrText_AS = orig.LocImpParentPkeyAttrText_AS;
      LocImpParentPkeyAttrText = orig.LocImpParentPkeyAttrText;
      LocExpParentPinstanceId_AS = orig.LocExpParentPinstanceId_AS;
      LocExpParentPinstanceId = orig.LocExpParentPinstanceId;
      LocExpParentPreferenceId_AS = orig.LocExpParentPreferenceId_AS;
      LocExpParentPreferenceId = orig.LocExpParentPreferenceId;
      LocExpParentPcreateUserId_AS = orig.LocExpParentPcreateUserId_AS;
      LocExpParentPcreateUserId = orig.LocExpParentPcreateUserId;
      LocExpParentPupdateUserId_AS = orig.LocExpParentPupdateUserId_AS;
      LocExpParentPupdateUserId = orig.LocExpParentPupdateUserId;
      LocExpParentPkeyAttrText_AS = orig.LocExpParentPkeyAttrText_AS;
      LocExpParentPkeyAttrText = orig.LocExpParentPkeyAttrText;
      LocExpParentPsearchAttrText_AS = orig.LocExpParentPsearchAttrText_AS;
      LocExpParentPsearchAttrText = orig.LocExpParentPsearchAttrText;
      LocExpParentPotherAttrText_AS = orig.LocExpParentPotherAttrText_AS;
      LocExpParentPotherAttrText = orig.LocExpParentPotherAttrText;
      LocExpParentPtypeTkeyAttrText_AS = orig.LocExpParentPtypeTkeyAttrText_AS;
      LocExpParentPtypeTkeyAttrText = orig.LocExpParentPtypeTkeyAttrText;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static MYY10121_LA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new MYY10121_LA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new MYY10121_LA());
          }
          else 
          {
            MYY10121_LA result = freeArray[--countFree];
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
      return(new MYY10121_LA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      LocImpParentPinstanceId_AS = ' ';
      LocImpParentPinstanceId = "00000000000000000000";
      LocImpParentPkeyAttrText_AS = ' ';
      LocImpParentPkeyAttrText = "     ";
      LocExpParentPinstanceId_AS = ' ';
      LocExpParentPinstanceId = "00000000000000000000";
      LocExpParentPreferenceId_AS = ' ';
      LocExpParentPreferenceId = "00000000000000000000";
      LocExpParentPcreateUserId_AS = ' ';
      LocExpParentPcreateUserId = "        ";
      LocExpParentPupdateUserId_AS = ' ';
      LocExpParentPupdateUserId = "        ";
      LocExpParentPkeyAttrText_AS = ' ';
      LocExpParentPkeyAttrText = "     ";
      LocExpParentPsearchAttrText_AS = ' ';
      LocExpParentPsearchAttrText = "                         ";
      LocExpParentPotherAttrText_AS = ' ';
      LocExpParentPotherAttrText = "                         ";
      LocExpParentPtypeTkeyAttrText_AS = ' ';
      LocExpParentPtypeTkeyAttrText = "    ";
    }
  }
}
