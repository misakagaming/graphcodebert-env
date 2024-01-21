// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: CYYY0111_IA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:40:48
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
  /// Internal data view storage for: CYYY0111_IA
  /// </summary>
  [Serializable]
  public class CYYY0111_IA : ViewBase, IImportView
  {
    private static CYYY0111_IA[] freeArray = new CYYY0111_IA[30];
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
    //        Type: PARENT
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpParentPkeyAttrText
    /// </summary>
    private char _ImpParentPkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpParentPkeyAttrText
    /// </summary>
    public char ImpParentPkeyAttrText_AS {
      get {
        return(_ImpParentPkeyAttrText_AS);
      }
      set {
        _ImpParentPkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpParentPkeyAttrText
    /// Domain: Text
    /// Length: 5
    /// Varying Length: N
    /// </summary>
    private string _ImpParentPkeyAttrText;
    /// <summary>
    /// Attribute for: ImpParentPkeyAttrText
    /// </summary>
    public string ImpParentPkeyAttrText {
      get {
        return(_ImpParentPkeyAttrText);
      }
      set {
        _ImpParentPkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpParentPsearchAttrText
    /// </summary>
    private char _ImpParentPsearchAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpParentPsearchAttrText
    /// </summary>
    public char ImpParentPsearchAttrText_AS {
      get {
        return(_ImpParentPsearchAttrText_AS);
      }
      set {
        _ImpParentPsearchAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpParentPsearchAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _ImpParentPsearchAttrText;
    /// <summary>
    /// Attribute for: ImpParentPsearchAttrText
    /// </summary>
    public string ImpParentPsearchAttrText {
      get {
        return(_ImpParentPsearchAttrText);
      }
      set {
        _ImpParentPsearchAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpParentPotherAttrText
    /// </summary>
    private char _ImpParentPotherAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpParentPotherAttrText
    /// </summary>
    public char ImpParentPotherAttrText_AS {
      get {
        return(_ImpParentPotherAttrText_AS);
      }
      set {
        _ImpParentPotherAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpParentPotherAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _ImpParentPotherAttrText;
    /// <summary>
    /// Attribute for: ImpParentPotherAttrText
    /// </summary>
    public string ImpParentPotherAttrText {
      get {
        return(_ImpParentPotherAttrText);
      }
      set {
        _ImpParentPotherAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpParentPtypeTkeyAttrText
    /// </summary>
    private char _ImpParentPtypeTkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpParentPtypeTkeyAttrText
    /// </summary>
    public char ImpParentPtypeTkeyAttrText_AS {
      get {
        return(_ImpParentPtypeTkeyAttrText_AS);
      }
      set {
        _ImpParentPtypeTkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpParentPtypeTkeyAttrText
    /// Domain: Text
    /// Length: 4
    /// Varying Length: N
    /// </summary>
    private string _ImpParentPtypeTkeyAttrText;
    /// <summary>
    /// Attribute for: ImpParentPtypeTkeyAttrText
    /// </summary>
    public string ImpParentPtypeTkeyAttrText {
      get {
        return(_ImpParentPtypeTkeyAttrText);
      }
      set {
        _ImpParentPtypeTkeyAttrText = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public CYYY0111_IA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public CYYY0111_IA( CYYY0111_IA orig )
    {
      ImpReferenceIyy1ServerDataUserid_AS = orig.ImpReferenceIyy1ServerDataUserid_AS;
      ImpReferenceIyy1ServerDataUserid = orig.ImpReferenceIyy1ServerDataUserid;
      ImpReferenceIyy1ServerDataReferenceId_AS = orig.ImpReferenceIyy1ServerDataReferenceId_AS;
      ImpReferenceIyy1ServerDataReferenceId = orig.ImpReferenceIyy1ServerDataReferenceId;
      ImpParentPkeyAttrText_AS = orig.ImpParentPkeyAttrText_AS;
      ImpParentPkeyAttrText = orig.ImpParentPkeyAttrText;
      ImpParentPsearchAttrText_AS = orig.ImpParentPsearchAttrText_AS;
      ImpParentPsearchAttrText = orig.ImpParentPsearchAttrText;
      ImpParentPotherAttrText_AS = orig.ImpParentPotherAttrText_AS;
      ImpParentPotherAttrText = orig.ImpParentPotherAttrText;
      ImpParentPtypeTkeyAttrText_AS = orig.ImpParentPtypeTkeyAttrText_AS;
      ImpParentPtypeTkeyAttrText = orig.ImpParentPtypeTkeyAttrText;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static CYYY0111_IA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new CYYY0111_IA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new CYYY0111_IA());
          }
          else 
          {
            CYYY0111_IA result = freeArray[--countFree];
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
      return(new CYYY0111_IA(this));
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
      ImpParentPkeyAttrText_AS = ' ';
      ImpParentPkeyAttrText = "     ";
      ImpParentPsearchAttrText_AS = ' ';
      ImpParentPsearchAttrText = "                         ";
      ImpParentPotherAttrText_AS = ' ';
      ImpParentPotherAttrText = "                         ";
      ImpParentPtypeTkeyAttrText_AS = ' ';
      ImpParentPtypeTkeyAttrText = "    ";
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
      this.CopyFrom((CYYY0111_IA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( CYYY0111_IA orig )
    {
      ImpReferenceIyy1ServerDataUserid_AS = orig.ImpReferenceIyy1ServerDataUserid_AS;
      ImpReferenceIyy1ServerDataUserid = orig.ImpReferenceIyy1ServerDataUserid;
      ImpReferenceIyy1ServerDataReferenceId_AS = orig.ImpReferenceIyy1ServerDataReferenceId_AS;
      ImpReferenceIyy1ServerDataReferenceId = orig.ImpReferenceIyy1ServerDataReferenceId;
      ImpParentPkeyAttrText_AS = orig.ImpParentPkeyAttrText_AS;
      ImpParentPkeyAttrText = orig.ImpParentPkeyAttrText;
      ImpParentPsearchAttrText_AS = orig.ImpParentPsearchAttrText_AS;
      ImpParentPsearchAttrText = orig.ImpParentPsearchAttrText;
      ImpParentPotherAttrText_AS = orig.ImpParentPotherAttrText_AS;
      ImpParentPotherAttrText = orig.ImpParentPotherAttrText;
      ImpParentPtypeTkeyAttrText_AS = orig.ImpParentPtypeTkeyAttrText_AS;
      ImpParentPtypeTkeyAttrText = orig.ImpParentPtypeTkeyAttrText;
    }
  }
}